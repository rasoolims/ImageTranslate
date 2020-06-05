import copy
import datetime
import os
import random
import sys
import time
from optparse import OptionParser
from typing import Dict

import torch
import torch.utils.data as data_utils
from IPython.core import ultratb
from torch.nn.utils.rnn import pad_sequence

import dataset
from albert_seq2seq import MassSeq2Seq
from lm import LM
from seq_gen import BeamDecoder, get_outputs_until_eos
from textprocessor import TextProcessor
from train_mt import MTTrainer

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


def mask_text(mask_prob, pads, texts, text_processor: TextProcessor):
    src_text = texts.clone()

    pad_indices = ((pads.cumsum(0) == pads) & pads).max(1)[1]
    interval_sizes = pad_indices / 2
    mask_indices = [random.randint(1, int(x)) for x in pad_indices - (1 - mask_prob) * pad_indices]
    src_mask = torch.zeros(src_text.size(), dtype=torch.bool)
    to_recover = []
    to_recover_pos = []
    for i, mask_start in enumerate(mask_indices):
        src_mask[i, mask_start: mask_start + int(pad_indices[i] / 2)] = True
        to_recover.append(torch.cat([src_text[i, 0:1], src_text[i, mask_start: mask_start + int(pad_indices[i] / 2)]]))
        to_recover_pos.append(
            torch.cat([torch.arange(0, 1), torch.arange(mask_start, mask_start + int(pad_indices[i] / 2))]))
    to_recover = pad_sequence(to_recover, batch_first=True, padding_value=text_processor.pad_token_id())
    to_recover_pos = pad_sequence(to_recover_pos, batch_first=True, padding_value=int(src_text.size(-1)) - 1)

    assert 0 < mask_prob < 1
    tgt_mask = ~src_mask
    tgt_mask[~pads] = False  # We should not mask pads.
    tgt_mask[:, 0] = False  # Always unmask the first token (start symbol or language identifier).

    replacements = src_text[src_mask]
    for i in range(len(replacements)):
        r = random.random()
        if r < 0.8:
            replacements[i] = text_processor.mask_token_id()
        elif r < 0.9:
            # Replace with another random word.
            random_index = random.randint(len(text_processor.special_tokens), text_processor.vocab_size() - 1)
            replacements[i] = random_index
        else:
            # keep the word
            pass
    src_text[src_mask] = replacements
    masked_ids = texts[:, 1:][src_mask[:, 1:]]

    return src_mask, masked_ids, src_text, to_recover, to_recover_pos


class MassTrainer(MTTrainer):
    def __init__(self, model: MassSeq2Seq, mask_prob: float = 0.5, clip: int = 1, optimizer=None,
                 warmup: int = 12500, step: int = 125000, fp16: bool = False, fp16_opt_level: str = "01",
                 beam_width: int = 5, max_len_a: float = 1.1, max_len_b: int = 5, len_penalty_ratio: float = 0.8):
        super().__init__(model=model, mask_prob=mask_prob, clip=clip, optimizer=optimizer, warmup=warmup, step=step,
                         fp16=fp16, fp16_opt_level=fp16_opt_level, beam_width=beam_width, max_len_a=max_len_a,
                         max_len_b=max_len_b, len_penalty_ratio=len_penalty_ratio)
        self.generator = BeamDecoder(model, beam_width=beam_width, max_len_a=max_len_a, max_len_b=max_len_b,
                                     len_penalty_ratio=len_penalty_ratio)

    def train_epoch(self, data_iter: data_utils.DataLoader, valid_data_iter: data_utils.DataLoader, saving_path: str,
                    step: int, max_grad_norm: float = 1.0, **kwargs):
        if self.fp16:
            try:
                import apex
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        sentences = 0
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        for i, batch in enumerate(data_iter):
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            src_inputs = batch["src_texts"].squeeze(0)
            src_pad_mask = batch["src_pad_mask"].squeeze(0)

            src_mask, targets, src_text, to_recover, positions = mask_text(self.mask_prob, src_pad_mask, src_inputs,
                                                                           model.text_processor)

            if src_inputs.size(0) < self.num_gpu:
                continue

            try:
                predictions = self.model(device=self.device, src_inputs=src_text, tgt_inputs=to_recover,
                                         tgt_positions=positions, src_pads=src_pad_mask,
                                         pad_idx=model.text_processor.pad_token_id(), log_softmax=True)
                ntokens = targets.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue

                loss = self.criterion(predictions, targets).mean()
                if self.fp16:
                    with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss = float(loss.data) * ntokens
                total_loss += loss
                cur_loss += loss
                total_tokens += ntokens
                tokens += ntokens
                sentences += int(src_inputs.size(0))

                if self.optimizer is not None:
                    # We accumulate the gradients for both tasks!
                    if self.fp16:
                        torch.nn.utils.clip_grad_norm_(apex.amp.master_params(self.optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    step += 1

            except RuntimeError as err:
                print("Error in processing", src_inputs.size(), src_inputs.size())
                torch.cuda.empty_cache()

            if step % 50 == 0 and tokens > 0:
                elapsed = time.time() - start
                print(datetime.datetime.now(),
                      "Epoch Step: %d Loss: %f Tokens per Sec: %f Sentences per Sec: %f" % (
                          step, cur_loss / tokens, tokens / elapsed, sentences / elapsed))

                if step % 1000 == 0:
                    # Save every 1000 steps!
                    model.save_checkpoint(saving_path)

                if step % 500 == 0:
                    self.validate(valid_data_iter)

                start, tokens, cur_loss, sentences = time.time(), 0, 0, 0

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        model.save(saving_path + ".latest")

        self.validate(valid_data_iter)
        return step

    def fine_tune(self, data_iter: data_utils.DataLoader, lang_directions: Dict[int, int], saving_path: str,
                  step: int, max_grad_norm: float = 1.0, valid_data_iter: data_utils.DataLoader = None):
        if self.fp16:
            try:
                import apex
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        sentences = 0
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        for i, batch in enumerate(data_iter):
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            src_inputs = batch["src_texts"].squeeze(0)
            src_pad_mask = batch["src_pad_mask"].squeeze(0)
            target_langs = torch.LongTensor([lang_directions[int(l)] for l in src_inputs[:, 0]])
            if src_inputs.size(0) < self.num_gpu:
                continue

            try:
                with torch.no_grad():
                    # We do not backpropagate the data generator following the MASS paper.
                    model.eval()
                    outputs = self.generator(device=self.device, src_inputs=src_inputs, tgt_langs=target_langs,
                                             src_mask=src_pad_mask)
                    translations = pad_sequence(outputs, batch_first=True)
                    translation_pad_mask = (translations != model.text_processor.pad_token_id())
                    model.train()

                # Now use it for back-translation loss.
                predictions = self.model(device=self.device, src_inputs=translations, tgt_inputs=src_inputs,
                                         src_mask=translation_pad_mask, tgt_mask=src_pad_mask,
                                         mask_pad_mask=src_pad_mask,  # Just pads for mask.
                                         log_softmax=True)
                src_targets = src_inputs[:, 1:].contiguous().view(-1)
                src_mask_flat = src_pad_mask[:, 1:].contiguous().view(-1)
                targets = src_targets[src_mask_flat]
                ntokens = targets.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue

                bt_loss = self.criterion(predictions, targets).mean()
                if self.fp16:
                    with apex.amp.scale_loss(bt_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    bt_loss.backward()

                bt_loss = float(bt_loss.data) * ntokens
                total_loss += bt_loss
                cur_loss += bt_loss
                total_tokens += ntokens
                tokens += ntokens
                sentences += int(src_inputs.size(0))

                if self.optimizer is not None:
                    # We accumulate the gradients for both tasks!
                    if self.fp16:
                        torch.nn.utils.clip_grad_norm_(apex.amp.master_params(self.optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    step += 1

            except RuntimeError as err:
                print("Error in processing", src_inputs.size(), src_inputs.size())
                torch.cuda.empty_cache()

            if step % 50 == 0 and tokens > 0:
                elapsed = time.time() - start
                print(datetime.datetime.now(),
                      "Epoch Step: %d Loss: %f Tokens per Sec: %f Sentences per Sec: %f" % (
                          step, cur_loss / tokens, tokens / elapsed, sentences / elapsed))

                if step % 1000 == 0:
                    # Save every 1000 steps!
                    model.save_checkpoint(saving_path)

                if step % 500 == 0 and valid_data_iter is not None:
                    bleu = self.eval_bleu(valid_data_iter, saving_path)
                    print("BLEU:", bleu)

                start, tokens, cur_loss, sentences = time.time(), 0, 0, 0

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        model.save(saving_path + ".latest")

        if valid_data_iter is not None:
            bleu = self.eval_bleu(valid_data_iter, saving_path)
            print("BLEU:", bleu)
        return step

    def validate(self, valid_data_iter):
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()
        with torch.no_grad():
            total_valid_loss, total_valid_tokens = 0, 0
            for batch in valid_data_iter:
                src_inputs = batch["src_texts"].squeeze(0)
                src_pad_mask = batch["src_pad_mask"].squeeze(0)

                src_mask, targets, src_text, to_recover, positions = mask_text(self.mask_prob, src_pad_mask, src_inputs,
                                                                               model.text_processor)

                try:
                    predictions = self.model(device=self.device, src_inputs=src_text, tgt_inputs=to_recover,
                                             tgt_positions=positions, src_pads=src_pad_mask,
                                             pad_idx=model.text_processor.pad_token_id(), log_softmax=True)
                    ntokens = targets.size(0)

                    if ntokens == 0:  # Nothing to predict!
                        continue

                    loss = self.criterion(predictions, targets).mean().data * ntokens
                    total_valid_loss += float(loss)
                    total_valid_tokens += ntokens
                except RuntimeError:
                    print("Error in processing", src_inputs.size(), src_inputs.size())
                    torch.cuda.empty_cache()

            valid_loss = total_valid_loss / total_valid_tokens
            print("Current valid loss", valid_loss)
            model.train()

    @staticmethod
    def train(options):
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)

        if options.pretrained_path is not None:
            mt_model, lm = MassSeq2Seq.load(options.pretrained_path)
        else:
            if options.lm_path is None:
                lm = LM(text_processor=text_processor, size=options.model_size)
            else:
                lm = LM.load(options.lm_path)

            decoder = copy.DeepCopy(lm.encoder) if options.sep_encoder else lm.encoder
            mt_model = MassSeq2Seq(config=lm.config, encoder=lm.encoder, decoder=decoder, output_layer=lm.masked_lm,
                                   text_processor=lm.text_processor, checkpoint=options.checkpoint)

        mt_model.save_config_and_tok(options.model_path)
        pin_memory = torch.cuda.is_available()

        train_loader, valid_loader, finetune_loader, mt_valid_loader = None, None, None, None
        if options.step > 0:
            train_data = dataset.MassDataset(batch_pickle_dir=options.train_path,
                                             max_batch_capacity=options.total_capacity, max_batch=options.batch,
                                             pad_idx=mt_model.text_processor.pad_token_id(),
                                             max_seq_len=options.max_seq_len)

            valid_data = dataset.MassDataset(batch_pickle_dir=options.valid_path,
                                             max_batch_capacity=options.total_capacity,
                                             max_batch=options.batch,
                                             pad_idx=mt_model.text_processor.pad_token_id(),
                                             max_seq_len=options.max_seq_len)
            train_loader = data_utils.DataLoader(train_data, batch_size=1, shuffle=True, pin_memory=pin_memory)
            valid_loader = data_utils.DataLoader(valid_data, batch_size=1, shuffle=False, pin_memory=pin_memory)

        lang_directions = {}
        if options.finetune_step > 0:
            finetune_data = dataset.MassDataset(batch_pickle_dir=options.train_path,
                                                max_batch_capacity=int(options.batch / (options.beam_width * 2)),
                                                max_batch=int(options.batch / (options.beam_width * 4)),
                                                pad_idx=mt_model.text_processor.pad_token_id(),
                                                max_seq_len=options.max_seq_len)
            finetune_loader = data_utils.DataLoader(finetune_data, batch_size=1, shuffle=True, pin_memory=pin_memory)
            for lang1 in finetune_data.lang_ids:
                for lang2 in finetune_data.lang_ids:
                    if lang1 != lang2:
                        # Assuming that we only have two languages!
                        lang_directions[lang1] = lang2

        trainer = MassTrainer(model=mt_model, mask_prob=options.mask_prob,
                              optimizer=MassTrainer.build_optimizer(mt_model, options.learning_rate,
                                                                    options.weight_decay),
                              clip=options.clip, warmup=options.warmup, step=options.step + options.finetune_step,
                              fp16=options.fp16,
                              fp16_opt_level=options.fp16_opt_level, beam_width=options.beam_width,
                              max_len_a=options.max_len_a, max_len_b=options.max_len_b,
                              len_penalty_ratio=options.len_penalty_ratio)

        mt_valid_loader = None
        if options.mt_valid_path is not None:
            mt_valid_data = dataset.MTDataset(batch_pickle_dir=options.mt_valid_path,
                                              max_batch_capacity=options.total_capacity,
                                              max_batch=int(options.batch / (options.beam_width * 2)),
                                              pad_idx=mt_model.text_processor.pad_token_id())
            mt_valid_loader = data_utils.DataLoader(mt_valid_data, batch_size=1, shuffle=False, pin_memory=pin_memory)

            print("creating reference")
            trainer.reference = []

            for batch in mt_valid_loader:
                tgt_inputs = batch["dst_texts"].squeeze()
                refs = get_outputs_until_eos(text_processor.sep_token_id(), tgt_inputs)
                ref = [trainer.generator.seq2seq_model.text_processor.tokenizer.decode(ref.numpy()) for ref in refs]
                trainer.reference += ref

        step, train_epoch = 0, 1

        while options.step > 0 and step <= options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(data_iter=train_loader, valid_data_iter=valid_loader,
                                       saving_path=options.model_path,
                                       step=step)
            train_epoch += 1

        finetune_epoch = 0
        while options.finetune_step > 0 and step <= options.finetune_step + options.step:
            print("finetune epoch", finetune_epoch)
            _ = trainer.fine_tune(data_iter=finetune_loader, lang_directions=lang_directions,
                                  saving_path=options.model_path, step=step, valid_data_iter=mt_valid_loader)
            finetune_epoch += 1


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--train", dest="train_path", help="Path to the train data pickle files", metavar="FILE",
                      default=None)
    parser.add_option("--valid", dest="valid_path",
                      help="Path to the dev data pickle files", metavar="FILE", default=None)
    parser.add_option("--valid_mt", dest="mt_valid_path",
                      help="Path to the MT dev data pickle files (SHOULD NOT BE USED IN UNSUPERVISED SETTING)",
                      metavar="FILE", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--model", dest="model_path", help="Directory path to save the best model", metavar="FILE",
                      default=None)
    parser.add_option("--lm", dest="lm_path", help="LM pretrained model", metavar="FILE", default=None)
    parser.add_option("--pretrained", dest="pretrained_path", help="MT pretrained model", metavar="FILE", default=None)
    parser.add_option("--clip", dest="clip", help="For gradient clipping", type="int", default=1)
    parser.add_option("--capacity", dest="total_capacity", help="Batch capcity", type="int", default=150)
    parser.add_option("--batch", dest="batch", help="Batch num_tokens", type="int", default=20000)
    parser.add_option("--mask", dest="mask_prob", help="Random masking probability", type="float", default=0.5)
    parser.add_option("--embed", dest="d_model", help="Embedding of contextual word vectors", type="int", default=768)
    parser.add_option("--lr", dest="learning_rate", help="Learning rate", type="float", default=0.002)
    parser.add_option("--warmup", dest="warmup", help="Number of warmup steps", type="int", default=12500)
    parser.add_option("--step", dest="step", help="Number of training steps", type="int", default=75000)
    parser.add_option("--fstep", dest="finetune_step", help="Number of finetuneing steps", type="int", default=75000)
    parser.add_option("--decay", dest="weight_decay", help="Weight decay", type="float", default=0.01)
    parser.add_option("--max_grad_norm", dest="max_grad_norm", help="Max grad norm", type="float", default=1.0)
    parser.add_option("--dropout", dest="dropout", help="Dropout probability", type="float", default=0.1)
    parser.add_option("--dff", dest="d_ff", help="Position-wise feed-forward dimensions", type="int", default=2048)
    parser.add_option("--layer", dest="num_layers", help="Number of Layers in cross-attention", type="int", default=2)
    parser.add_option("--beam", dest="beam_width", help="Beam width", type="int", default=5)
    parser.add_option("--max_seq_len", dest="max_seq_len", help="Max sequence length", type="int", default=175)
    parser.add_option("--heads", dest="num_heads", help="Number of attention heads", type="int", default=8)
    parser.add_option("--fp16", action="store_true", dest="fp16", help="use fp16; should be compatible", default=False)
    parser.add_option("--sep", action="store_true", dest="sep_encoder", help="Disjoint encoder/decoder", default=False)
    parser.add_option("--size", dest="model_size", help="1 base, 2 medium, 3 small, 4 toy", type="int", default=3)
    parser.add_option("--max_len_a", dest="max_len_a", help="a for beam search (a*l+b)", type="float", default=1.8)
    parser.add_option("--max_len_b", dest="max_len_b", help="b for beam search (a*l+b)", type="int", default=5)
    parser.add_option("--len-penalty", dest="len_penalty_ratio", help="Length penalty", type="float", default=0.8)
    parser.add_option("--checkpoint", dest="checkpoint", help="Number of checkpoints to average", type="int", default=5)
    parser.add_option(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    print(options)
    MassTrainer.train(options=options)
    print("Finished Training!")
