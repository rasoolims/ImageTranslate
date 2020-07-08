import copy
import datetime
import os
import pickle
import sys
import time
from typing import List

import sacrebleu
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import transformers.optimization as optim
from IPython.core import ultratb
from apex import amp

import dataset
from albert_seq2seq import AlbertSeq2Seq
from lm import LM
from loss import SmoothedNLLLoss
from option_parser import get_mt_option_parser
from parallel import DataParallelModel, DataParallelCriterion
from pytorch_lamb.pytorch_lamb import Lamb
from seq_gen import BeamDecoder, get_outputs_until_eos
from textprocessor import TextProcessor
from utils import build_optimizer, backward

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class MTTrainer:
    def __init__(self, model, mask_prob: float = 0.3, clip: int = 1, optimizer=None, warmup: int = 12500,
                 step: int = 125000, beam_width: int = 5, max_len_a: float = 1.1, max_len_b: int = 5,
                 len_penalty_ratio: float = 0.8, last_epoch: int = 0, nll_loss: bool = False, fp16: bool = False):
        self.model = model

        self.clip = clip
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if isinstance(self.optimizer, Lamb):
            self.scheduler = optim.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup,
                                                                   num_training_steps=step)
            self.scheduler.last_epoch = last_epoch
            print("Scheduler Last epoch", last_epoch)
        else:
            self.scheduler = None

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

        self.fp16 = False
        if self.num_gpu == 1 and fp16:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O2")
            self.fp16 = True

        self.reference = None
        self.best_bleu = -1.0

    def train_epoch(self, data_iter: List[data_utils.DataLoader], dev_data_iter: List[data_utils.DataLoader],
                    saving_path: str, step: int):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        sentences = 0
        model = self.model.module if hasattr(self.model, "module") else self.model

        shortest = min([len(l) for l in data_iter])
        data_to_iter = zip(data_iter[0], data_iter[1]) if len(data_iter) == 2 else zip(data_iter[0])
        for i, batches in enumerate(data_to_iter):
            for batch in batches:
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                src_inputs = batch["src_texts"].squeeze(0)
                src_mask = batch["src_pad_mask"].squeeze(0)
                tgt_inputs = batch["dst_texts"].squeeze(0)
                tgt_mask = batch["dst_pad_mask"].squeeze(0)
                src_langs = batch["src_langs"].squeeze(0)
                dst_langs = batch["dst_langs"].squeeze(0)
                if src_inputs.size(0) < self.num_gpu:
                    continue

                try:
                    predictions = self.model(src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                             src_mask=src_mask, tgt_mask=tgt_mask, src_langs=src_langs,
                                             tgt_langs=dst_langs,
                                             log_softmax=True)
                    targets = tgt_inputs[:, 1:].contiguous().view(-1)
                    tgt_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
                    targets = targets[tgt_mask_flat]
                    ntokens = targets.size(0)
                    if self.num_gpu == 1:
                        targets = targets.to(predictions.device)

                    if ntokens == 0:  # Nothing to predict!
                        continue

                    loss = self.criterion(predictions, targets).mean()
                    backward(loss, self.optimizer, self.fp16)

                    loss = float(loss.data) * ntokens
                    total_loss += loss
                    cur_loss += loss
                    total_tokens += ntokens
                    tokens += ntokens
                    sentences += int(src_inputs.size(0))
                    if self.optimizer is not None:
                        # We accumulate the gradients for both tasks!
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                        self.optimizer.step()
                        if self.scheduler is not None:
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
                        model.save(saving_path)
                        with open(os.path.join(saving_path, "optim"), "wb") as fp:
                            pickle.dump(
                                (self.optimizer, self.scheduler.last_epoch if self.scheduler is not None else step),
                                fp)

                    if step % 500 == 0:
                        self.validate(dev_data_iter)
                        bleu = self.eval_bleu(dev_data_iter, saving_path)
                        print("BLEU:", bleu)

                    start, tokens, cur_loss, sentences = time.time(), 0, 0, 0

            if i == shortest:
                break

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        model.save(saving_path + ".latest")
        with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
            pickle.dump((self.optimizer, self.scheduler.last_epoch if self.scheduler is not None else step), fp)

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

        with torch.no_grad():
            for iter in dev_data_iter:
                for batch in iter:
                    src_inputs = batch["src_texts"].squeeze(0)
                    src_mask = batch["src_pad_mask"].squeeze(0)
                    tgt_inputs = batch["dst_texts"].squeeze(0)
                    src_langs = batch["src_langs"].squeeze(0)
                    dst_langs = batch["dst_langs"].squeeze(0)
                    src_pad_idx = batch["pad_idx"].squeeze(0)

                    src_ids = get_outputs_until_eos(model.text_processor.sep_token_id(), src_inputs,
                                                    remove_first_token=True)
                    src_text += list(map(lambda src: model.text_processor.tokenizer.decode(src.numpy()), src_ids))

                    outputs = self.generator(src_inputs=src_inputs, src_sizes=src_pad_idx,
                                             first_tokens=tgt_inputs[:, 0],
                                             src_mask=src_mask, src_langs=src_langs, tgt_langs=dst_langs,
                                             pad_idx=model.text_processor.pad_token_id())
                    if self.num_gpu > 1:
                        new_outputs = []
                        for output in outputs:
                            new_outputs += output
                        outputs = new_outputs

                    mt_output += list(map(lambda x: model.text_processor.tokenizer.decode(x[1:].numpy()), outputs))

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
                pickle.dump((self.optimizer, self.scheduler.last_epoch if self.scheduler is not None else 0), fp)

            with open(os.path.join(saving_path, "bleu.best.output"), "w") as writer:
                writer.write("\n".join(
                    [src + "\n" + ref + "\n" + o + "\n\n***************\n" for src, ref, o in
                     zip(src_text, mt_output, self.reference[:len(mt_output)])]))

        return bleu.score

    def validate(self, dev_data_iters):
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()
        with torch.no_grad():
            total_dev_loss, total_dev_tokens = 0, 0
            for dev_data_iter in dev_data_iters:
                for batch in dev_data_iter:
                    src_inputs = batch["src_texts"].squeeze(0)
                    src_mask = batch["src_pad_mask"].squeeze(0)
                    tgt_inputs = batch["dst_texts"].squeeze(0)
                    tgt_mask = batch["dst_pad_mask"].squeeze(0)
                    src_langs = batch["src_langs"].squeeze(0)
                    dst_langs = batch["dst_langs"].squeeze(0)

                    try:
                        predictions = self.model(src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                                 src_mask=src_mask, tgt_mask=tgt_mask, src_langs=src_langs,
                                                 tgt_langs=dst_langs, log_softmax=True)

                        targets = tgt_inputs[:, 1:].contiguous().view(-1)
                        tgt_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
                        targets = targets[tgt_mask_flat]
                        if self.num_gpu == 1:
                            targets = targets.to(predictions.device)
                        ntokens = targets.size(0)

                        if ntokens == 0:  # Nothing to predict!
                            continue

                        loss = self.criterion(predictions, targets).mean().data * ntokens
                        total_dev_loss += float(loss)
                        total_dev_tokens += ntokens
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
                                              sep_decoder=options.sep_encoder, lang_dec=options.lang_decoder)
        else:
            if options.lm_path is None:
                lm = LM(text_processor=text_processor, size=options.model_size)
            else:
                lm = LM.load(options.lm_path)

            encoder = copy.deepcopy(lm.encoder) if options.sep_encoder else lm.encoder
            mt_model = AlbertSeq2Seq(config=lm.config, encoder=encoder, decoder=lm.encoder, output_layer=lm.masked_lm,
                                     text_processor=lm.text_processor, checkpoint=options.checkpoint,
                                     lang_dec=options.lang_decoder)

        mt_model.save(options.model_path)
        train_paths = options.train_path.strip().split(",")
        pin_memory = torch.cuda.is_available()
        dev_paths = options.dev_path.strip().split(",")
        train_loader, dev_loader = [], []
        num_processors = max(torch.cuda.device_count(), 1)
        for train_path in train_paths:
            train_data = dataset.MTDataset(batch_pickle_dir=train_path,
                                           max_batch_capacity=num_processors * options.total_capacity,
                                           max_batch=num_processors * options.batch,
                                           pad_idx=mt_model.text_processor.pad_token_id())

            tl = data_utils.DataLoader(train_data, batch_size=1, shuffle=True, pin_memory=pin_memory)
            train_loader.append(tl)

        for dev_path in dev_paths:
            dev_data = dataset.MTDataset(batch_pickle_dir=dev_path,
                                         max_batch_capacity=num_processors * options.total_capacity,
                                         max_batch=num_processors * int(options.batch / options.beam_width),
                                         pad_idx=mt_model.text_processor.pad_token_id())
            dl = data_utils.DataLoader(dev_data, batch_size=1, shuffle=False, pin_memory=pin_memory)
            dev_loader.append(dl)

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer, last_epoch = pickle.load(fp)
        else:
            optimizer, last_epoch = build_optimizer(mt_model, options.learning_rate, options.weight_decay,
                                                    use_adam=options.adam), 0
        trainer = MTTrainer(model=mt_model, mask_prob=options.mask_prob, optimizer=optimizer, clip=options.clip,
                            warmup=options.warmup, step=options.step, beam_width=options.beam_width,
                            max_len_a=options.max_len_a, max_len_b=options.max_len_b,
                            len_penalty_ratio=options.len_penalty_ratio, fp16=options.fp16,
                            last_epoch=last_epoch, nll_loss=options.nll_loss)

        print("creating reference")
        trainer.reference = []
        generator = (
            trainer.generator.module if hasattr(trainer.generator, "module") else trainer.generator
        )
        for dl in dev_loader:
            for batch in dl:
                tgt_inputs = batch["dst_texts"].squeeze()
                refs = get_outputs_until_eos(text_processor.sep_token_id(), tgt_inputs, remove_first_token=True)
                ref = [generator.seq2seq_model.text_processor.tokenizer.decode(ref.numpy()) for ref in refs]
                trainer.reference += ref

        step, train_epoch = 0, 1
        while step <= options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(data_iter=train_loader, dev_data_iter=dev_loader,
                                       saving_path=options.model_path,
                                       step=step)
            train_epoch += 1

    @staticmethod
    def config_dropout(mt_model, dropout):
        mt_model.encoder.config.hidden_dropout_prob = dropout
        mt_model.encoder.config.attention_probs_dropout_prob = dropout
        if isinstance(mt_model.decoder, nn.ModuleList):
            for i, dec in enumerate(mt_model.decoder):
                dec.config.hidden_dropout_prob = dropout
                dec.config.attention_probs_dropout_prob = dropout
        else:
            mt_model.decoder.config.hidden_dropout_prob = dropout
            mt_model.decoder.config.attention_probs_dropout_prob = dropout


if __name__ == "__main__":
    parser = get_mt_option_parser()
    (options, args) = parser.parse_args()
    print(options)
    MTTrainer.train(options=options)
    print("Finished Training!")
