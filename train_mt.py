import copy
import datetime
import os
import pickle
import sys
import time
from typing import Optional

import sacrebleu
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import transformers.optimization as optim
from IPython.core import ultratb

import dataset
import train_lm
from albert_seq2seq import AlbertSeq2Seq
from lm import LM
from loss import SmoothedNLLLoss
from parallel import DataParallelModel, DataParallelCriterion
from seq_gen import BeamDecoder, get_outputs_until_eos
from textprocessor import TextProcessor

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class MTTrainer:
    def __init__(self, model, mask_prob: float = 0.3, clip: int = 1, optimizer=None, warmup: int = 12500,
                 step: int = 125000, beam_width: int = 5, max_len_a: float = 1.1, max_len_b: int = 5,
                 len_penalty_ratio: float = 0.8, self_translate: bool = False, last_epoch: int = 0,
                 nll_loss: bool = False):
        self.model = model

        self.clip = clip
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.self_translate = self_translate

        self.scheduler = optim.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup,
                                                               num_training_steps=step)
        self.scheduler.last_epoch = last_epoch
        print("Scheduler Last epoch", last_epoch)

        self.mask_prob = mask_prob
        if nll_loss:
            self.criterion = nn.NLLLoss(ignore_index=model.text_processor.pad_token_id())
        else:
            self.criterion = SmoothedNLLLoss(ignore_index=model.text_processor.pad_token_id())

        self.num_gpu = torch.cuda.device_count()
        if self.num_gpu > 1:
            print("Let's use", self.num_gpu, "GPUs!")
            self.model = DataParallelModel(self.model)
            self.criterion = DataParallelCriterion(self.criterion)

        self.generator = BeamDecoder(model, beam_width=beam_width, max_len_a=max_len_a, max_len_b=max_len_b,
                                     len_penalty_ratio=len_penalty_ratio)
        if self.num_gpu > 1:
            self.generator = DataParallelModel(self.generator)

        self.reference = None
        self.best_bleu = -1.0

    def train_epoch(self, data_iter: data_utils.DataLoader, dev_data_iter: data_utils.DataLoader, saving_path: str,
                    step: int, max_grad_norm: float = 1.0,
                    monolingual_data_iter: Optional[data_utils.DataLoader] = None):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        sentences = 0
        model = self.model.module if hasattr(self.model, "module") else self.model

        data_to_iter = data_iter if monolingual_data_iter is None else zip(data_iter, monolingual_data_iter)
        for i, batched in enumerate(data_to_iter):
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            batch = batched if monolingual_data_iter is None else batched[0]
            src_inputs = batch["src_texts"].squeeze(0)
            src_mask = batch["src_pad_mask"].squeeze(0)
            tgt_inputs = batch["dst_texts"].squeeze(0)
            tgt_mask = batch["dst_pad_mask"].squeeze(0)
            src_langs = batch["src_langs"]
            dst_langs = batch["dst_langs"]
            if src_inputs.size(0) < self.num_gpu:
                continue

            try:
                if self.self_translate:
                    mask, masked_ids, src_inputs = LM.mask_text(mask_prob=self.mask_prob, pads=src_mask,
                                                                texts=src_inputs,
                                                                text_processor=model.text_processor,
                                                                mask_eos=False)

                predictions = self.model(device=self.device, src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                         src_mask=src_mask, tgt_mask=tgt_mask, src_langs=src_langs, tgt_langs=dst_langs,
                                         log_softmax=True)
                targets = tgt_inputs[:, 1:].contiguous().view(-1)
                tgt_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
                targets = targets[tgt_mask_flat]
                ntokens = targets.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue

                loss = self.criterion(predictions, targets).mean()
                loss.backward()

                loss = float(loss.data) * ntokens
                total_loss += loss
                cur_loss += loss
                total_tokens += ntokens
                tokens += ntokens
                sentences += int(src_inputs.size(0))
                if self.self_translate:
                    LM.unmask_text(mask=mask, masked_ids=masked_ids, texts=src_inputs)

                if monolingual_data_iter is not None:
                    src_inputs = batched[1]["src_texts"].squeeze(0)
                    src_mask = batched[1]["src_pad_mask"].squeeze(0)
                    tgt_inputs = batched[1]["dst_texts"].squeeze(0)
                    tgt_mask = batched[1]["dst_pad_mask"].squeeze(0)
                    src_langs = batch["src_langs"]
                    dst_langs = batch["dst_langs"]

                    mask, masked_ids, src_inputs = LM.mask_text(mask_prob=self.mask_prob, pads=src_mask,
                                                                texts=src_inputs,
                                                                text_processor=model.text_processor,
                                                                mask_eos=False)

                    predictions = self.model(device=self.device, src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                             src_mask=src_mask, tgt_mask=tgt_mask, src_langs=src_langs,
                                             tgt_langs=dst_langs, log_softmax=True)
                    targets = tgt_inputs[:, 1:].contiguous().view(-1)
                    tgt_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
                    targets = targets[tgt_mask_flat]
                    ntokens = targets.size(0)

                    if ntokens == 0:  # Nothing to predict!
                        continue

                    loss = self.criterion(predictions, targets).mean()
                    loss.backward()

                    loss = float(loss.data) * ntokens
                    total_loss += loss
                    cur_loss += loss
                    total_tokens += ntokens
                    tokens += ntokens
                    sentences += int(src_inputs.size(0))
                    LM.unmask_text(mask=mask, masked_ids=masked_ids, texts=src_inputs)

                if self.optimizer is not None:
                    # We accumulate the gradients for both tasks!
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    step += 1


            except RuntimeError as err:
                print("Error in processing", src_inputs.size(), tgt_inputs.size())
                torch.cuda.empty_cache()

            if step % 50 == 0 and tokens > 0:
                elapsed = time.time() - start
                print(datetime.datetime.now(),
                      "Epoch Step: %d Loss: %f Tokens per Sec: %f Sentences per Sec: %f" % (
                          step, cur_loss / tokens, tokens / elapsed, sentences / elapsed))

                if step % 1000 == 0:
                    # Save every 1000 steps!
                    model.save_checkpoint(saving_path)
                    with open(os.path.join(saving_path, "optim"), "wb") as fp:
                        pickle.dump((self.optimizer, self.scheduler.last_epoch), fp)

                if step % 500 == 0:
                    self.validate(dev_data_iter)
                    bleu = self.eval_bleu(dev_data_iter, saving_path)
                    print("BLEU:", bleu)

                start, tokens, cur_loss, sentences = time.time(), 0, 0, 0

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        model.save(saving_path + ".latest")
        with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
            pickle.dump((self.optimizer, self.scheduler.last_epoch), fp)

        self.validate(dev_data_iter)
        bleu = self.eval_bleu(dev_data_iter, saving_path)
        print("BLEU:", bleu)
        return step

    def eval_bleu(self, dev_data_iter, saving_path):
        mt_output = []
        src_text = []
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()
        generator = (
            self.generator.module if hasattr(self.generator, "module") else self.generator
        )

        with torch.no_grad():
            for batch in dev_data_iter:
                src_inputs = batch["src_texts"].squeeze(0)
                src_mask = batch["src_pad_mask"].squeeze(0)
                tgt_inputs = batch["dst_texts"].squeeze(0)
                src_langs = batch["src_langs"].squeeze(0)
                dst_langs = batch["dst_langs"].squeeze(0)

                if self.self_translate:
                    mask, masked_ids, src_inputs = LM.mask_text(mask_prob=0.15, pads=src_mask, texts=src_inputs,
                                                                text_processor=model.text_processor, mask_eos=False)

                src_ids = get_outputs_until_eos(model.text_processor.sep_token_id(), src_inputs)
                src_text += [generator.seq2seq_model.text_processor.tokenizer.decode(src.numpy()) for src in src_ids]

                outputs = self.generator(device=self.device, src_inputs=src_inputs, first_tokens=tgt_inputs[:, 0],
                                         src_mask=src_mask, src_langs=src_langs, tgt_langs=dst_langs,
                                         pad_idx=model.text_processor.pad_token_id())
                if self.num_gpu > 1:
                    new_outputs = []
                    for output in outputs:
                        new_outputs += output
                    outputs = new_outputs

                for output in outputs:
                    mt_output.append(generator.seq2seq_model.text_processor.tokenizer.decode(output.numpy()))

                if self.self_translate:
                    LM.unmask_text(mask=mask, masked_ids=masked_ids, texts=src_inputs)

            model.train()
        bleu = sacrebleu.corpus_bleu(mt_output, [self.reference[:len(mt_output)]])

        with open(os.path.join(saving_path, "bleu.output"), "w") as writer:
            writer.write("\n".join(
                [src + "\n" + ref + "\n" + o + "\n\n***************\n" for src, ref, o in
                 zip(src_text, mt_output, self.reference[:len(mt_output)])]))

        if bleu.score > self.best_bleu:
            self.best_bleu = bleu.score
            print("Saving best BLEU", self.best_bleu)
            model.save(saving_path)
            with open(os.path.join(saving_path, "optim"), "wb") as fp:
                pickle.dump((self.optimizer, self.scheduler.last_epoch), fp)

            with open(os.path.join(saving_path, "bleu.best.output"), "w") as writer:
                writer.write("\n".join(
                    [src + "\n" + ref + "\n" + o + "\n\n***************\n" for src, ref, o in
                     zip(src_text, mt_output, self.reference[:len(mt_output)])]))

        return bleu.score

    def validate(self, dev_data_iter):
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()
        with torch.no_grad():
            total_dev_loss, total_dev_tokens = 0, 0
            for batch in dev_data_iter:
                src_inputs = batch["src_texts"].squeeze(0)
                src_mask = batch["src_pad_mask"].squeeze(0)
                tgt_inputs = batch["dst_texts"].squeeze(0)
                tgt_mask = batch["dst_pad_mask"].squeeze(0)
                src_langs = batch["src_langs"]
                dst_langs = batch["dst_langs"]

                try:
                    if self.self_translate:
                        mask, masked_ids, src_inputs = LM.mask_text(mask_prob=self.mask_prob, pads=src_mask,
                                                                    texts=src_inputs,
                                                                    text_processor=model.text_processor, mask_eos=False)

                    predictions = self.model(device=self.device, src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                             src_mask=src_mask, tgt_mask=tgt_mask, src_langs=src_langs,
                                             tgt_langs=dst_langs, log_softmax=True)

                    targets = tgt_inputs[:, 1:].contiguous().view(-1)
                    tgt_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
                    targets = targets[tgt_mask_flat]
                    ntokens = targets.size(0)

                    if ntokens == 0:  # Nothing to predict!
                        continue

                    loss = self.criterion(predictions, targets).mean().data * ntokens
                    total_dev_loss += float(loss)
                    total_dev_tokens += ntokens
                    if self.self_translate:
                        LM.unmask_text(mask=mask, masked_ids=masked_ids, texts=src_inputs)
                except RuntimeError:
                    print("Error in processing", src_inputs.size(), tgt_inputs.size())
                    torch.cuda.empty_cache()

            dev_loss = total_dev_loss / total_dev_tokens
            print("Current dev loss", dev_loss)
            model.train()

    @staticmethod
    def train(options):
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)

        if options.pretrained_path is not None:
            mt_model, lm = AlbertSeq2Seq.load(options.pretrained_path, tok_dir=options.tokenizer_path,
                                              sep_decoder=options.sep_encoder)
        else:
            if options.lm_path is None:
                lm = LM(text_processor=text_processor, size=options.model_size)
            else:
                lm = LM.load(options.lm_path)

            decoder = copy.deepcopy(lm.encoder) if options.sep_encoder else lm.encoder
            masked_lm = copy.deepcopy(lm.masked_lm) if options.sep_encoder else lm.masked_lm
            mt_model = AlbertSeq2Seq(config=lm.config, encoder=lm.encoder, decoder=decoder, output_layer=masked_lm,
                                     text_processor=lm.text_processor, checkpoint=options.checkpoint)

        mt_model.save_config_and_tok(options.model_path)

        train_data = dataset.MTDataset(batch_pickle_dir=options.train_path,
                                       max_batch_capacity=options.total_capacity, max_batch=options.batch,
                                       pad_idx=mt_model.text_processor.pad_token_id())
        dev_data = dataset.MTDataset(batch_pickle_dir=options.dev_path,
                                     max_batch_capacity=options.total_capacity,
                                     max_batch=int(options.batch / options.beam_width),
                                     pad_idx=mt_model.text_processor.pad_token_id())
        monolingual_data = None
        if options.monolingual_path is not None:
            monolingual_data = dataset.MTDataset(batch_pickle_dir=options.monolingual_path,
                                                 max_batch_capacity=options.total_capacity, max_batch=options.batch,
                                                 pad_idx=mt_model.text_processor.pad_token_id())

        pin_memory = torch.cuda.is_available()
        train_loader = data_utils.DataLoader(train_data, batch_size=1, shuffle=True, pin_memory=pin_memory)
        monolingual_loader = data_utils.DataLoader(monolingual_data, batch_size=1, shuffle=True,
                                                   pin_memory=pin_memory) if monolingual_data is not None else None
        dev_loader = data_utils.DataLoader(dev_data, batch_size=1, shuffle=False, pin_memory=pin_memory)

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer, last_epoch = pickle.load(fp)
        else:
            optimizer, last_epoch = train_lm.LMTrainer.build_optimizer(mt_model, options.learning_rate,
                                                                       options.weight_decay), 0
        trainer = MTTrainer(model=mt_model, mask_prob=options.mask_prob, optimizer=optimizer, clip=options.clip,
                            warmup=options.warmup, step=options.step, beam_width=options.beam_width,
                            max_len_a=options.max_len_a, max_len_b=options.max_len_b,
                            len_penalty_ratio=options.len_penalty_ratio, self_translate=options.pretrain,
                            last_epoch=last_epoch, nll_loss=options.nll_loss)

        print("creating reference")
        trainer.reference = []
        generator = (
            trainer.generator.module if hasattr(trainer.generator, "module") else trainer.generator
        )
        for batch in dev_loader:
            tgt_inputs = batch["dst_texts"].squeeze()
            refs = get_outputs_until_eos(text_processor.sep_token_id(), tgt_inputs)
            ref = [generator.seq2seq_model.text_processor.tokenizer.decode(ref.numpy()) for ref in refs]
            trainer.reference += ref

        print("Trying if largest batch fits into memory")
        MTTrainer.memory_test(train_data, trainer)

        step, train_epoch = 0, 1
        while step <= options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(data_iter=train_loader, dev_data_iter=dev_loader,
                                       monolingual_data_iter=monolingual_loader, saving_path=options.model_path,
                                       step=step)
            train_epoch += 1

    @staticmethod
    def config_dropout(mt_model, dropout):
        mt_model.encoder.config.hidden_dropout_prob = dropout
        mt_model.encoder.config.attention_probs_dropout_prob = dropout
        mt_model.decoder.config.hidden_dropout_prob = dropout
        mt_model.decoder.config.attention_probs_dropout_prob = dropout

    @staticmethod
    def memory_test(train_data, trainer):
        src_inputs = train_data.longest_batch[0]["src_texts"]
        src_mask = train_data.longest_batch[0]["src_pad_mask"]
        tgt_inputs = train_data.longest_batch[0]["dst_texts"]
        tgt_mask = train_data.longest_batch[0]["dst_pad_mask"]
        src_langs = train_data.longest_batch[0]["src_langs"]
        dst_langs = train_data.longest_batch[0]["dst_langs"]
        s, d, b = int(src_inputs.size(1)), int(tgt_inputs.size(1)), int(src_inputs.size(0))
        print(src_inputs.size(), tgt_inputs.size(), b * d * (s ** 2 + d ** 2))
        predictions = trainer.model(device=trainer.device, src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                    src_mask=src_mask, tgt_mask=tgt_mask, src_langs=src_langs, tgt_langs=dst_langs,
                                    log_softmax=True)
        targets = tgt_inputs[:, 1:].contiguous().view(-1)
        tgt_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
        targets = targets[tgt_mask_flat]

        ntokens = targets.size(0)
        if ntokens > 0:  # Nothing to predict!
            loss = trainer.criterion(predictions, targets).mean()
            loss.backward()
        trainer.optimizer.zero_grad()
        torch.cuda.empty_cache()

        src_inputs = train_data.most_token_batch[0]["src_texts"]
        src_mask = train_data.most_token_batch[0]["src_pad_mask"]
        tgt_inputs = train_data.most_token_batch[0]["dst_texts"]
        tgt_mask = train_data.most_token_batch[0]["dst_pad_mask"]
        src_langs = train_data.most_token_batch[0]["src_langs"]
        dst_langs = train_data.most_token_batch[0]["dst_langs"]
        s, d, b = int(src_inputs.size(1)), int(tgt_inputs.size(1)), int(src_inputs.size(0))
        print(src_inputs.size(), tgt_inputs.size(), b * (s + d))
        predictions = trainer.model(device=trainer.device, src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                    src_mask=src_mask, tgt_mask=tgt_mask, src_langs=src_langs, tgt_langs=dst_langs,
                                    log_softmax=True)
        targets = tgt_inputs[:, 1:].contiguous().view(-1)
        tgt_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
        targets = targets[tgt_mask_flat]
        ntokens = targets.size(0)
        if ntokens > 0:  # Nothing to predict!
            loss = trainer.criterion(predictions, targets).mean()
            loss.backward()
        trainer.optimizer.zero_grad()
        trainer.optimizer.step()
        trainer.scheduler.step()
        torch.cuda.empty_cache()


def get_option_parser():
    parser = train_lm.get_option_parser()
    parser.add_option("--mono", dest="monolingual_path",
                      help="Path to the monolingual data pickle files for auxiliary BART training", metavar="FILE",
                      default=None)
    parser.add_option("--capacity", dest="total_capacity", help="Batch capcity", type="int", default=150)
    parser.add_option("--lm", dest="lm_path", help="LM pretrained model", metavar="FILE", default=None)
    parser.add_option("--beam", dest="beam_width", help="Beam width", type="int", default=5)
    parser.add_option("--sep", action="store_true", dest="sep_encoder", help="Disjoint encoder/decoder", default=False)
    parser.add_option("--max_len_a", dest="max_len_a", help="a for beam search (a*l+b)", type="float", default=1.3)
    parser.add_option("--max_len_b", dest="max_len_b", help="b for beam search (a*l+b)", type="int", default=5)
    parser.add_option("--len-penalty", dest="len_penalty_ratio", help="Length penalty", type="float", default=0.8)
    parser.add_option("--checkpoint", dest="checkpoint", help="Number of checkpoints to average", type="int", default=5)
    parser.add_option("--max_seq_len", dest="max_seq_len", help="Max sequence length", type="int", default=175)
    parser.add_option("--pretrain", action="store_true", dest="pretrain",
                      help="Use self to self translation similar to BART!", default=False)
    parser.add_option("--nll", action="store_true", dest="nll_loss", help="Use NLL loss instead of smoothed NLL loss",
                      default=False)
    parser.set_default("batch", 20000)
    return parser


if __name__ == "__main__":
    parser = get_option_parser()
    (options, args) = parser.parse_args()
    print(options)
    MTTrainer.train(options=options)
    print("Finished Training!")
