import copy
import datetime
import os
import pickle
import sys
import time
from typing import Dict, List

import torch
import torch.utils.data as data_utils
import transformers.optimization as optim
from IPython.core import ultratb
from torch.nn.utils.rnn import pad_sequence

import dataset
from albert_seq2seq import MassSeq2Seq
from lm import LM
from option_parser import get_mass_option_parser
from seq_gen import get_outputs_until_eos
from textprocessor import TextProcessor
from train_mt import MTTrainer
from utils import build_optimizer, mass_mask, mass_unmask

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class MassTrainer(MTTrainer):
    def train_epoch(self, data_iter: List[data_utils.DataLoader], saving_path: str, step: int,
                    dev_data_iter: List[data_utils.DataLoader] = None, mt_dev_iter: List[data_utils.DataLoader] = None,
                    **kwargs):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        sentences = 0
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        shortest = min([len(l) for l in data_iter])
        # Here we assume that the data_iter has only two elements.
        for i, batches in enumerate(zip(data_iter[0], data_iter[1])):
            for batch in batches:
                self.optimizer.zero_grad()
                src_inputs = batch["src_texts"].squeeze(0)
                src_pad_mask = batch["src_pad_mask"].squeeze(0)
                pad_indices = batch["pad_idx"].squeeze(0)
                if src_inputs.size(0) < self.num_gpu:
                    continue

                masked_info = mass_mask(self.mask_prob, pad_indices, src_inputs, model.text_processor)
                try:
                    predictions = self.model(src_inputs=masked_info["src_text"],
                                             tgt_inputs=masked_info["to_recover"],
                                             tgt_positions=masked_info["positions"], src_pads=src_pad_mask,
                                             pad_idx=model.text_processor.pad_token_id(),
                                             src_langs=batch["langs"].squeeze(0),
                                             log_softmax=True)
                    ntokens = masked_info["targets"].size(0)

                    if ntokens == 0:  # Nothing to predict!
                        continue

                    targets = masked_info["targets"]
                    if self.num_gpu == 1:
                        targets = targets.to(predictions.device)

                    loss = self.criterion(predictions, targets).mean() * ntokens
                    loss.backward()

                    loss = float(loss.data)
                    total_loss += loss
                    cur_loss += loss
                    total_tokens += ntokens
                    tokens += ntokens
                    sentences += int(src_inputs.size(0))

                    # We accumulate the gradients for both tasks!
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    step += 1

                except RuntimeError as err:
                    torch.cuda.empty_cache()
                    print("Error in processing", src_inputs.size(), src_inputs.size())
                mass_unmask(masked_info["src_text"], masked_info["src_mask"], masked_info["mask_idx"])

                if step % 50 == 0 and tokens > 0:
                    elapsed = time.time() - start
                    print(datetime.datetime.now(),
                          "Epoch Step: %d Loss: %f Tokens per Sec: %f Sentences per Sec: %f" % (
                              step, cur_loss / tokens, tokens / elapsed, sentences / elapsed))

                    if step % 500 == 0:
                        # Save every 1000 steps!
                        model.save(saving_path + ".latest")
                        with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
                            pickle.dump(
                                (self.optimizer, self.scheduler.last_epoch if self.scheduler is not None else step), fp)

                    if step % 5000 == 0:
                        self.validate(dev_data_iter)
                        if mt_dev_iter is not None:
                            bleu = self.eval_bleu(mt_dev_iter, saving_path)
                            print("Pretraining BLEU:", bleu)

                    start, tokens, cur_loss, sentences = time.time(), 0, 0, 0
            if i == shortest - 1:
                break  # Visited all elements in one data!

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        model.save(saving_path + ".latest")
        with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
            pickle.dump((self.optimizer, self.scheduler.last_epoch if self.scheduler is not None else step), fp)

        self.validate(dev_data_iter)
        if mt_dev_iter is not None:
            bleu = self.eval_bleu(mt_dev_iter, saving_path)
            print("Pretraining BLEU:", bleu)
        return step

    def fine_tune(self, data_iter: List[data_utils.DataLoader], lang_directions: Dict[int, int], saving_path: str,
                  step: int, dev_data_iter: List[data_utils.DataLoader] = None, ):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        sentences = 0
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        shortest = min([len(l) for l in data_iter])
        # Here we assume that the data_iter has only two elements.
        for i, batches in enumerate(zip(data_iter[0], data_iter[1])):
            for batch in batches:
                self.optimizer.zero_grad()
                src_inputs = batch["src_texts"].squeeze(0)
                src_pad_mask = batch["src_pad_mask"].squeeze(0)
                src_pad_idx = batch["pad_idx"].squeeze(0)

                target_langs = torch.LongTensor([lang_directions[int(l)] for l in src_inputs[:, 0]])
                dst_langs = torch.LongTensor(
                    [model.text_processor.languages[model.text_processor.id2token(lang_directions[int(l)])] for l in
                     src_inputs[:, 0]])
                if src_inputs.size(0) < self.num_gpu:
                    continue

                try:
                    model.eval()
                    with torch.no_grad():
                        # We do not backpropagate the data generator following the MASS paper.
                        outputs = self.generator(src_inputs=src_inputs, src_sizes=src_pad_idx,
                                                 first_tokens=target_langs,
                                                 src_langs=batch["langs"].squeeze(0), tgt_langs=dst_langs,
                                                 pad_idx=model.text_processor.pad_token_id(),
                                                 src_mask=src_pad_mask, unpad_output=False, beam_width=1)
                        if self.num_gpu > 1:
                            new_outputs = []
                            for output in outputs:
                                new_outputs += output
                            outputs = new_outputs

                        translations = pad_sequence(outputs, batch_first=True)
                        translation_pad_mask = (translations != model.text_processor.pad_token_id())
                    model.train()

                    # Now use it for back-translation loss.
                    predictions = self.model(src_inputs=translations, tgt_inputs=src_inputs,
                                             src_pads=translation_pad_mask,
                                             pad_idx=model.text_processor.pad_token_id(),
                                             src_langs=dst_langs,
                                             tgt_langs=batch["langs"].squeeze(0),
                                             log_softmax=True)
                    src_targets = src_inputs[:, 1:].contiguous().view(-1)
                    src_mask_flat = src_pad_mask[:, 1:].contiguous().view(-1)
                    targets = src_targets[src_mask_flat]
                    ntokens = targets.size(0)

                    if ntokens == 0:  # Nothing to predict!
                        continue
                    if self.num_gpu == 1:
                        targets = targets.to(predictions.device)

                    bt_loss = self.criterion(predictions, targets).mean()
                    bt_loss.backward()

                    bt_loss = float(bt_loss.data) * ntokens
                    total_loss += bt_loss
                    cur_loss += bt_loss
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
                    torch.cuda.empty_cache()
                    print("Error in processing", src_inputs.size(), src_inputs.size())

                if step % 50 == 0 and tokens > 0:
                    elapsed = time.time() - start
                    print(datetime.datetime.now(),
                          "Epoch Step: %d Loss: %f Tokens per Sec: %f Sentences per Sec: %f" % (
                              step, cur_loss / tokens, tokens / elapsed, sentences / elapsed))

                    if step % 5000 == 0:
                        # Save every 1000 steps!
                        model.save(saving_path + ".beam.latest")
                        with open(os.path.join(saving_path + ".beam.latest", "optim"), "wb") as fp:
                            pickle.dump(
                                (self.optimizer, self.scheduler.last_epoch if self.scheduler is not None else step), fp)

                    if step % 5000 == 0 and dev_data_iter is not None:
                        bleu = self.eval_bleu(dev_data_iter, saving_path + ".beam")
                        print("BLEU:", bleu)

                    start, tokens, cur_loss, sentences = time.time(), 0, 0, 0
            if i == shortest - 1:
                break

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        model.save(saving_path + ".beam.latest")
        with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
            pickle.dump((self.optimizer, self.scheduler.last_epoch if self.scheduler is not None else step), fp)

        if dev_data_iter is not None:
            bleu = self.eval_bleu(dev_data_iter, saving_path + ".beam")
            print("BLEU:", bleu)
        return step

    def validate(self, dev_data_iters):
        if dev_data_iters is None:
            return
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()
        with torch.no_grad():
            total_dev_loss, total_dev_tokens = 0, 0
            for dev_data_iter in dev_data_iters:
                for batch in dev_data_iter:
                    src_inputs = batch["src_texts"].squeeze(0)
                    src_pad_mask = batch["src_pad_mask"].squeeze(0)
                    pad_indices = batch["pad_idx"].squeeze(0)

                    try:
                        masked_info = mass_mask(self.mask_prob, pad_indices, src_inputs, model.text_processor)
                        predictions = self.model(src_inputs=masked_info["src_text"],
                                                 tgt_inputs=masked_info["to_recover"],
                                                 tgt_positions=masked_info["positions"], src_pads=src_pad_mask,
                                                 pad_idx=model.text_processor.pad_token_id(),
                                                 src_langs=batch["langs"].squeeze(0),
                                                 log_softmax=True)
                        ntokens = masked_info["targets"].size(0)

                        if ntokens == 0:  # Nothing to predi`ct!
                            continue
                        targets = masked_info["targets"]
                        if self.num_gpu == 1:
                            targets = targets.to(predictions.device)

                        loss = self.criterion(predictions, targets).mean().data * ntokens
                        total_dev_loss += float(loss)
                        total_dev_tokens += ntokens
                    except RuntimeError:
                        torch.cuda.empty_cache()
                        print("Error in processing", src_inputs.size(), src_inputs.size())
                    mass_unmask(masked_info["src_text"], masked_info["src_mask"], masked_info["mask_idx"])

            dev_loss = total_dev_loss / total_dev_tokens
            print("Current dev loss", dev_loss)
            model.train()

    @staticmethod
    def train(options):
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        if options.pretrained_path is not None:
            mt_model, lm = MassSeq2Seq.load(out_dir=options.pretrained_path, tok_dir=options.tokenizer_path,
                                            sep_decoder=options.sep_encoder, lang_dec=options.lang_decoder)
            text_processor = mt_model.text_processor
        else:
            text_processor = TextProcessor(options.tokenizer_path)
            if options.lm_path is None:
                lm = LM(text_processor=text_processor, size=options.model_size)
            else:
                lm = LM.load(options.lm_path)

            encoder = copy.deepcopy(lm.encoder) if options.sep_encoder else lm.encoder
            mt_model = MassSeq2Seq(config=lm.config, encoder=encoder, decoder=lm.encoder, output_layer=lm.masked_lm,
                                   text_processor=lm.text_processor, lang_dec=options.lang_decoder,
                                   checkpoint=options.checkpoint)
        MTTrainer.config_dropout(mt_model, options.dropout)

        num_processors = max(torch.cuda.device_count(), 1)
        pin_memory = torch.cuda.is_available()

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer, last_epoch = pickle.load(fp)
        else:
            optimizer, last_epoch = build_optimizer(mt_model, options.learning_rate, options.weight_decay,
                                                    use_adam=options.adam), 0

        train_data, train_loader, dev_loader, finetune_loader, mt_dev_loader = None, None, None, None, None
        train_paths = options.train_path.strip().split(",")
        if options.step > 0 and last_epoch < options.step:
            train_data, train_loader = [], []
            for i, train_path in enumerate(train_paths):
                td = dataset.MassDataset(batch_pickle_dir=train_path,
                                         max_batch_capacity=num_processors * options.total_capacity,
                                         max_batch=num_processors * options.batch,
                                         pad_idx=mt_model.text_processor.pad_token_id(),
                                         max_seq_len=options.max_seq_len, keep_examples=True)
                train_data.append(td)
                dl = data_utils.DataLoader(td, batch_size=1, shuffle=True, pin_memory=pin_memory)
                train_loader.append(dl)

            if options.dev_path is not None:
                dev_paths = options.dev_path.strip().split(",")
                dev_loader = list()
                for dev_path in dev_paths:
                    dev_data = dataset.MassDataset(batch_pickle_dir=dev_path,
                                                   max_batch_capacity=num_processors * options.total_capacity,
                                                   max_batch=num_processors * options.batch,
                                                   pad_idx=mt_model.text_processor.pad_token_id(),
                                                   max_seq_len=options.max_seq_len)
                    dl = data_utils.DataLoader(dev_data, batch_size=1, shuffle=False, pin_memory=pin_memory)
                    dev_loader.append(dl)

        lang_directions = {}
        if options.finetune_step > 0:
            finetune_data, finetune_loader = [], []
            for i, train_path in enumerate(train_paths):
                fd = dataset.MassDataset(batch_pickle_dir=train_path,
                                         max_batch_capacity=num_processors * int(options.total_capacity / 2),
                                         max_batch=num_processors * int(options.batch / 2),
                                         pad_idx=mt_model.text_processor.pad_token_id(),
                                         max_seq_len=options.max_seq_len, keep_examples=False,
                                         example_list=None if train_data is None else train_data[i].examples_list)
                finetune_data.append(fd)
                fl = data_utils.DataLoader(fd, batch_size=1, shuffle=True, pin_memory=pin_memory)
                finetune_loader.append(fl)

                if train_data is not None:
                    train_data[i].examples_list = []

            langs = set()
            for fd in finetune_data:
                for lang1 in fd.lang_ids:
                    langs.add(lang1)

            for lang1 in langs:
                for lang2 in langs:
                    if lang1 != lang2:
                        # Assuming that we only have two languages!
                        lang_directions[lang1] = lang2

        trainer = MassTrainer(model=mt_model, mask_prob=options.mask_prob, optimizer=optimizer, clip=options.clip,
                              warmup=options.warmup, step=options.step if options.step > 0 else options.finetune_step,
                              beam_width=options.beam_width, max_len_a=options.max_len_a, max_len_b=options.max_len_b,
                              len_penalty_ratio=options.len_penalty_ratio, last_epoch=last_epoch,
                              nll_loss=options.nll_loss)

        mt_dev_loader = None
        if options.mt_dev_path is not None:
            mt_dev_loader = []
            dev_paths = options.mt_dev_path.split(",")
            trainer.reference = []
            for dev_path in dev_paths:
                mt_dev_data = dataset.MTDataset(batch_pickle_dir=dev_path,
                                                max_batch_capacity=num_processors * options.total_capacity,
                                                max_batch=num_processors * int(
                                                    options.batch / (options.beam_width * 2)),
                                                pad_idx=mt_model.text_processor.pad_token_id())
                dl = data_utils.DataLoader(mt_dev_data, batch_size=1, shuffle=False, pin_memory=pin_memory)
                mt_dev_loader.append(dl)

                print("creating reference")

                generator = (
                    trainer.generator.module if hasattr(trainer.generator, "module") else trainer.generator
                )

                for batch in dl:
                    tgt_inputs = batch["dst_texts"].squeeze()
                    refs = get_outputs_until_eos(text_processor.sep_token_id(), tgt_inputs, remove_first_token=True)
                    ref = [generator.seq2seq_model.text_processor.tokenizer.decode(ref.numpy()) for ref in refs]
                    trainer.reference += ref

        step, train_epoch = last_epoch, 0

        while options.step > 0 and step < options.step:
            train_epoch += 1
            print("train epoch", train_epoch)
            step = trainer.train_epoch(data_iter=train_loader, dev_data_iter=dev_loader,
                                       saving_path=options.model_path, mt_dev_iter=mt_dev_loader,
                                       step=step)

        finetune_epoch = 0
        mt_model.save(options.model_path + ".beam")
        if train_epoch > 0:
            # Resetting the optimizer for the purpose of finetuning.
            model = mt_model.module if hasattr(mt_model, "module") else mt_model
            trainer.optimizer = build_optimizer(model, options.learning_rate, options.weight_decay,
                                                use_adam=options.adam)
            if not options.adam:
                trainer.scheduler = optim.get_linear_schedule_with_warmup(trainer.optimizer,
                                                                          num_warmup_steps=options.warmup,
                                                                          num_training_steps=options.finetune_step)

        while options.finetune_step > 0 and step <= options.finetune_step + options.step:
            print("finetune epoch", finetune_epoch)
            step = trainer.fine_tune(data_iter=finetune_loader, lang_directions=lang_directions,
                                     saving_path=options.model_path, step=step, dev_data_iter=mt_dev_loader)
            finetune_epoch += 1


if __name__ == "__main__":
    parser = get_mass_option_parser()
    (options, args) = parser.parse_args()
    print(options)
    MassTrainer.train(options=options)
    print("Finished Training!")
