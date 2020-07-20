import copy
import datetime
import os
import pickle
import random
import sys
import time
from itertools import chain
from typing import List

import sacrebleu
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from IPython.core import ultratb
from apex import amp
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

import dataset
from image_doc_model import ImageMassSeq2Seq
from lm import LM
from loss import SmoothedNLLLoss
from option_parser import get_img_options_parser
from parallel import DataParallelModel, DataParallelCriterion
from seq_gen import BeamDecoder, get_outputs_until_eos
from textprocessor import TextProcessor
from utils import build_optimizer, mass_mask, mass_unmask, backward

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class ImageDocTrainer:
    def __init__(self, model, mask_prob: float = 0.3, clip: int = 1, optimizer=None,
                 beam_width: int = 5, max_len_a: float = 1.1, max_len_b: int = 5,
                 len_penalty_ratio: float = 0.8, nll_loss: bool = False, fp16: bool = False, mm_mode="mixed"):
        self.model = model

        self.clip = clip
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.num_gpu = torch.cuda.device_count()

        self.mask_prob = mask_prob
        if nll_loss:
            self.criterion = nn.NLLLoss(ignore_index=model.text_processor.pad_token_id())
        else:
            self.criterion = SmoothedNLLLoss(ignore_index=model.text_processor.pad_token_id())

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
        self.mm_mode = mm_mode

    def train_epoch(self, img_data_iter: List[data_utils.DataLoader], step: int, saving_path: str = None,
                    mass_data_iter: List[data_utils.DataLoader] = None, mt_dev_iter: List[data_utils.DataLoader] = None,
                    mt_train_iter: List[data_utils.DataLoader] = None, max_step: int = 300000,
                    fine_tune: bool = False, lang_directions: dict = False, **kwargs):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        batch_zip, shortest = self.get_batch_zip(img_data_iter, mass_data_iter, mt_train_iter)

        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        for i, batches in enumerate(batch_zip):
            for batch in batches:
                self.optimizer.zero_grad()
                is_img_batch = isinstance(batch, list) and "captions" in batch[0]
                is_mass_batch = not is_img_batch and "dst_texts" not in batch
                is_contrastive = False
                if True:
                    if fine_tune and (is_img_batch or is_mass_batch):
                        id2lid = lambda r: model.text_processor.languages[
                            model.text_processor.id2token(lang_directions[int(r)])]
                        if is_mass_batch:
                            src_inputs = batch["src_texts"].squeeze(0)
                            src_pad_mask = batch["src_pad_mask"].squeeze(0)
                            pad_indices = batch["pad_idx"].squeeze(0)
                            target_langs = torch.LongTensor([lang_directions[int(l)] for l in src_inputs[:, 0]])
                            dst_langs = torch.LongTensor([id2lid(l) for l in src_inputs[:, 0]])
                        else:
                            src_inputs = [b["captions"] for b in batch]
                            src_pad_mask = [b["caption_mask"] for b in batch]
                            pad_indices = [b["pad_idx"] for b in batch]
                            target_langs = [torch.LongTensor([lang_directions[int(l)] for l in src[:, 0]]) for src in
                                            src_inputs]
                            dst_langs = [torch.LongTensor([id2lid(l) for l in src[:, 0]]) for src in src_inputs]
                        if len(src_inputs) < self.num_gpu:
                            continue

                        if is_mass_batch:
                            langs = batch["langs"].squeeze(0)
                        else:
                            langs = [b["langs"] for b in batch]

                        model.eval()
                        with torch.no_grad():
                            # We do not backpropagate the data generator following the MASS paper.
                            images = None
                            if is_img_batch:
                                images = [b["images"] for b in batch]
                            outputs = self.generator(src_inputs=src_inputs,
                                                     src_sizes=pad_indices,
                                                     first_tokens=target_langs,
                                                     src_langs=langs, tgt_langs=dst_langs,
                                                     pad_idx=model.text_processor.pad_token_id(),
                                                     src_mask=src_pad_mask, unpad_output=False, beam_width=1,
                                                     images=images)
                            if self.num_gpu > 1:
                                if is_mass_batch:
                                    new_outputs = []
                                    for output in outputs:
                                        new_outputs += output
                                    outputs = new_outputs

                            if is_mass_batch or self.num_gpu <= 1:
                                translations = pad_sequence(outputs, batch_first=True,
                                                            padding_value=model.text_processor.pad_token_id())
                                translation_pad_mask = (translations != model.text_processor.pad_token_id())
                            else:
                                translations = [pad_sequence(output, batch_first=True,
                                                             padding_value=model.text_processor.pad_token_id()) for
                                                output in outputs]
                                translation_pad_mask = [t != model.text_processor.pad_token_id() for t in translations]
                        model.train()

                        if is_mass_batch:
                            langs = batch["langs"].squeeze(0)
                        else:
                            langs = torch.cat([b["langs"] for b in batch])
                        # Now use it for back-translation loss.
                        predictions = self.model(src_inputs=translations,
                                                 tgt_inputs=src_inputs,
                                                 src_pads=translation_pad_mask,
                                                 pad_idx=model.text_processor.pad_token_id(),
                                                 src_langs=dst_langs,
                                                 tgt_langs=langs,
                                                 log_softmax=True)
                        if is_mass_batch:
                            src_targets = src_inputs[:, 1:].contiguous().view(-1)
                            src_mask_flat = src_pad_mask[:, 1:].contiguous().view(-1)
                        else:
                            src_targets = torch.cat(list(map(lambda s: s[:, 1:], src_inputs)))
                            src_mask_flat = torch.cat(list(map(lambda s: s[:, 1:], src_pad_mask)))
                        targets = src_targets[src_mask_flat]

                        ntokens = targets.size(0)
                    elif is_img_batch:
                        src_inputs = [b["captions"] for b in batch]
                        src_pad_mask = [b["caption_mask"] for b in batch]
                        langs = [b["langs"] for b in batch]
                        if (self.mm_mode == "mixed" and random.random() <= .5) or self.mm_mode == "masked":
                            pad_indices = [b["pad_idx"] for b in batch]
                            if len(batch) < self.num_gpu:
                                continue

                            masked_info = list(
                                map(lambda pi, si: mass_mask(self.mask_prob, pi, si, model.text_processor), pad_indices,
                                    src_inputs))
                            predictions = self.model(src_inputs=list(map(lambda m: m["src_text"], masked_info)),
                                                     tgt_inputs=list(map(lambda m: m["to_recover"], masked_info)),
                                                     tgt_positions=list(map(lambda m: m["positions"], masked_info)),
                                                     src_pads=src_pad_mask,
                                                     pad_idx=model.text_processor.pad_token_id(),
                                                     src_langs=langs, batch=batch,
                                                     log_softmax=True)
                            targets = torch.cat(list(map(lambda m: m["targets"], masked_info)))
                            ntokens = targets.size(0)
                        else:
                            neg_samples = [b["neg"] for b in batch]
                            neg_mask = [b["neg_mask"] for b in batch]
                            loss = self.model(src_inputs=src_inputs,
                                              src_pads=src_pad_mask,
                                              neg_samples=neg_samples,
                                              neg_mask=neg_mask,
                                              pad_idx=model.text_processor.pad_token_id(),
                                              src_langs=langs, batch=batch,
                                              log_softmax=True)
                            is_contrastive = True

                    elif not is_mass_batch:  # MT data
                        src_inputs = batch["src_texts"].squeeze(0)
                        src_mask = batch["src_pad_mask"].squeeze(0)
                        tgt_inputs = batch["dst_texts"].squeeze(0)
                        tgt_mask = batch["dst_pad_mask"].squeeze(0)
                        src_langs = batch["src_langs"].squeeze(0)
                        dst_langs = batch["dst_langs"].squeeze(0)
                        if src_inputs.size(0) < self.num_gpu:
                            continue
                        predictions = self.model(src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                                 src_pads=src_mask, tgt_mask=tgt_mask, src_langs=src_langs,
                                                 tgt_langs=dst_langs, log_softmax=True)
                        targets = tgt_inputs[:, 1:].contiguous().view(-1)
                        tgt_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
                        targets = targets[tgt_mask_flat]
                        ntokens = targets.size(0)
                    else:  # MASS data
                        src_inputs = batch["src_texts"].squeeze(0)
                        src_pad_mask = batch["src_pad_mask"].squeeze(0)
                        pad_indices = batch["pad_idx"].squeeze(0)
                        if src_inputs.size(0) < self.num_gpu:
                            continue

                        masked_info = mass_mask(self.mask_prob, pad_indices, src_inputs, model.text_processor)
                        predictions = self.model(src_inputs=masked_info["src_text"],
                                                 tgt_inputs=masked_info["to_recover"],
                                                 tgt_positions=masked_info["positions"], src_pads=src_pad_mask,
                                                 pad_idx=model.text_processor.pad_token_id(),
                                                 src_langs=batch["langs"].squeeze(0),
                                                 log_softmax=True)
                        targets = masked_info["targets"]
                        ntokens = targets.size(0)

                    if is_contrastive:  # Nothing to predict!
                        backward(loss, self.optimizer, self.fp16)
                        loss = loss.data
                    elif ntokens > 0:
                        if self.num_gpu == 1:
                            targets = targets.to(predictions.device)

                        loss = self.criterion(predictions, targets).mean()
                        backward(loss, self.optimizer, self.fp16)

                        loss = float(loss.data) * ntokens
                        tokens += ntokens
                        total_tokens += ntokens
                    total_loss += loss
                    cur_loss += loss

                    # We accumulate the gradients for both tasks!
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    step += 1

                    if is_mass_batch and not fine_tune:
                        mass_unmask(masked_info["src_text"], masked_info["src_mask"], masked_info["mask_idx"])
                    if not is_contrastive and is_img_batch and not fine_tune:
                        map(lambda m: mass_unmask(m["src_text"], m["src_mask"], m["mask_idx"]), masked_info)

                else:
                    print(repr(err))
                    print("Error processing", is_img_batch)
                    if (isinstance(model, ImageMassSeq2Seq)) and is_img_batch:
                        for b in batch:
                            print("->", len(b["images"]), b["captions"].size())
                    torch.cuda.empty_cache()

                if step % 50 == 0 and tokens > 0:
                    elapsed = time.time() - start
                    print(datetime.datetime.now(),
                          "Epoch Step: %d Loss: %f Tokens per Sec: %f " % (step, cur_loss / tokens, tokens / elapsed))

                    if step % 500 == 0:
                        print(model.multimodal_attention_gate)
                        if mt_dev_iter is not None and step % 5000 == 0:
                            bleu = self.eval_bleu(mt_dev_iter, saving_path)
                            print("BLEU:", bleu)

                        model.save(saving_path + ".latest")
                        with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
                            pickle.dump(self.optimizer, fp)

                    start, tokens, cur_loss = time.time(), 0, 0
            if i == shortest - 1:
                break

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        model.save(saving_path + ".latest")

        if mt_dev_iter is not None:
            print(model.multimodal_attention_gate)
            bleu = self.eval_bleu(mt_dev_iter, saving_path)
            print("BLEU:", bleu)

        return step

    def get_batch_zip(self, img_data_iter, mass_data_iter, mt_train_iter):
        if img_data_iter is not None and mt_train_iter is not None:
            img_data_iter *= 5
        if mass_data_iter is not None and mt_train_iter is not None:
            mass_data_iter *= 5
        iters = list(chain(*filter(lambda x: x != None, [img_data_iter, mass_data_iter, mt_train_iter])))
        shortest = min(len(l) for l in iters)
        return zip(*iters), shortest

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
        bleu = sacrebleu.corpus_bleu(mt_output, [self.reference[:len(mt_output)]], lowercase=True, tokenize="intl")

        with open(os.path.join(saving_path, "bleu.output"), "w") as writer:
            writer.write("\n".join(
                [src + "\n" + ref + "\n" + o + "\n\n***************\n" for src, ref, o in
                 zip(src_text, mt_output, self.reference[:len(mt_output)])]))

        if bleu.score > self.best_bleu:
            self.best_bleu = bleu.score
            print("Saving best BLEU", self.best_bleu)
            model.save(saving_path)
            with open(os.path.join(saving_path, "optim"), "wb") as fp:
                pickle.dump(self.optimizer, fp)

            with open(os.path.join(saving_path, "bleu.best.output"), "w") as writer:
                writer.write("\n".join(
                    [src + "\n" + ref + "\n" + o + "\n\n***************\n" for src, ref, o in
                     zip(src_text, mt_output, self.reference[:len(mt_output)])]))

        return bleu.score

    @staticmethod
    def train(options):
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)
        num_processors = max(torch.cuda.device_count(), 1)

        if options.pretrained_path is not None:
            mt_model, lm = ImageMassSeq2Seq.load(options.pretrained_path, tok_dir=options.tokenizer_path,
                                                 sep_decoder=options.sep_encoder, resnet_depth=options.resnet_depth,
                                                 lang_dec=options.lang_decoder)
        else:
            if options.lm_path is None:
                lm = LM(text_processor=text_processor, size=options.model_size)
            else:
                lm = LM.load(options.lm_path)

            decoder = copy.deepcopy(lm.encoder) if options.sep_encoder else lm.encoder
            mt_model = ImageMassSeq2Seq(config=lm.config, encoder=lm.encoder, decoder=decoder,
                                        output_layer=lm.masked_lm,
                                        text_processor=lm.text_processor, checkpoint=options.checkpoint,
                                        resnet_depth=options.resnet_depth, lang_dec=options.lang_decoder)

        transform = transforms.Compose([  # [1]
            transforms.Resize(256),  # [2]
            transforms.CenterCrop(224),  # [3]
            transforms.ToTensor(),  # [4]
            transforms.Normalize(  # [5]
                mean=[0.485, 0.456, 0.406],  # [6]
                std=[0.229, 0.224, 0.225]  # [7]
            )])

        print("Model initialization done!")

        # We assume that the collator function returns a list with the size of number of gpus (in case of cpus,
        collator = dataset.ImageTextCollator()
        num_batches = max(1, torch.cuda.device_count())

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer = pickle.load(fp)
        else:
            optimizer = build_optimizer(mt_model, options.learning_rate, warump_steps=options.warmup)
        trainer = ImageDocTrainer(model=mt_model, mask_prob=options.mask_prob, optimizer=optimizer, clip=options.clip,
                                  beam_width=options.beam_width, max_len_a=options.max_len_a,
                                  max_len_b=options.max_len_b, len_penalty_ratio=options.len_penalty_ratio,
                                  fp16=options.fp16, mm_mode=options.mm_mode)

        pin_memory = torch.cuda.is_available()
        img_train_loader = None
        img_train_loader = ImageDocTrainer.get_img_loader(collator, dataset.ImageCaptionDataset, img_train_loader,
                                                          mt_model, num_batches, options, pin_memory, transform)

        mass_train_data, mass_train_loader, finetune_loader, mt_dev_loader = None, None, None, None
        if options.mass_train_path is not None:
            mass_train_paths = options.mass_train_path.strip().split(",")
            if options.step > 0:
                mass_train_data, mass_train_loader = ImageDocTrainer.get_mass_loader(mass_train_paths, mt_model,
                                                                                     num_processors, options,
                                                                                     pin_memory)

            if options.finetune_step > 0:
                finetune_loader, finetune_data = ImageDocTrainer.get_mass_finetune_data(mass_train_data,
                                                                                        mass_train_paths, mt_model,
                                                                                        num_processors, options,
                                                                                        pin_memory)

        mt_train_loader = None
        if options.mt_train_path is not None:
            mt_train_loader = ImageDocTrainer.get_mt_train_data(mt_model, num_processors, options, pin_memory)

        mt_dev_loader = None
        if options.mt_dev_path is not None:
            mt_dev_loader = ImageDocTrainer.get_mt_dev_data(mt_model, options, pin_memory, text_processor, trainer)

        step, train_epoch = 0, 1
        while options.step > 0 and step < options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(img_data_iter=img_train_loader, mass_data_iter=mass_train_loader,
                                       mt_train_iter=mt_train_loader, max_step=options.step,
                                       mt_dev_iter=mt_dev_loader, saving_path=options.model_path, step=step)
            train_epoch += 1

        finetune_epoch = 0
        mt_model.save(options.model_path + ".beam")
        # Resetting the optimizer for the purpose of finetuning.
        trainer.optimizer.reset()

        lang_directions = ImageDocTrainer.get_lang_dirs(options.bt_langs, text_processor)

        print("Reloading image train data with new batch size...")
        if options.finetune_step > 0 and img_train_loader is not None:
            img_train_loader = ImageDocTrainer.get_img_loader(collator, dataset.ImageCaptionDataset, img_train_loader,
                                                              mt_model, num_batches, options, pin_memory,
                                                              transform, 2)
        print("Reloading image train data with new batch size done!")

        while options.finetune_step > 0 and step <= options.finetune_step + options.step:
            print("finetune epoch", finetune_epoch)
            step = trainer.train_epoch(img_data_iter=img_train_loader, mass_data_iter=finetune_loader,
                                       mt_train_iter=mt_train_loader, max_step=options.finetune_step + options.step,
                                       mt_dev_iter=mt_dev_loader, saving_path=options.model_path, step=step,
                                       fine_tune=True, lang_directions=lang_directions)
            finetune_epoch += 1

    @staticmethod
    def get_lang_dirs(bt_langs, text_processor: TextProcessor):
        langs = ["<" + l + ">" for l in bt_langs.strip().split(",")]
        langs = set([text_processor.token_id(l) for l in langs])
        if len(langs) < 2:
            return None
        assert len(langs) <= 2
        lang_directions = {}
        for lang1 in langs:
            for lang2 in langs:
                if lang1 != lang2:
                    # Assuming that we only have two languages!
                    lang_directions[lang1] = lang2
        return lang_directions

    @staticmethod
    def get_mt_dev_data(mt_model, options, pin_memory, text_processor, trainer):
        mt_dev_loader = []
        dev_paths = options.mt_dev_path.split(",")
        trainer.reference = []
        for dev_path in dev_paths:
            mt_dev_data = dataset.MTDataset(batch_pickle_dir=dev_path,
                                            max_batch_capacity=options.total_capacity,
                                            max_batch=int(options.batch / (options.beam_width * 2)),
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
        return mt_dev_loader

    @staticmethod
    def get_mt_train_data(mt_model, num_processors, options, pin_memory):
        mt_train_loader = []
        train_paths = options.mt_train_path.split(",")
        for train_path in train_paths:
            mt_train_data = dataset.MTDataset(batch_pickle_dir=train_path,
                                              max_batch_capacity=int(num_processors * options.total_capacity / 2),
                                              max_batch=int(num_processors * options.batch / 2),
                                              pad_idx=mt_model.text_processor.pad_token_id())
            mtl = data_utils.DataLoader(mt_train_data, batch_size=1, shuffle=True, pin_memory=pin_memory)
            mt_train_loader.append(mtl)
        return mt_train_loader

    @staticmethod
    def get_mass_finetune_data(mass_train_data, mass_train_paths, mt_model, num_processors, options, pin_memory):
        finetune_data, finetune_loader = [], []
        for i, mass_train_path in enumerate(mass_train_paths):
            fd = dataset.MassDataset(batch_pickle_dir=mass_train_path,
                                     max_batch_capacity=int(num_processors * options.total_capacity / 2),
                                     max_batch=int(num_processors * options.batch / 2),
                                     pad_idx=mt_model.text_processor.pad_token_id(),
                                     max_seq_len=options.max_seq_len, keep_examples=False,
                                     example_list=None if mass_train_data is None else mass_train_data[
                                         i].examples_list)
            finetune_data.append(fd)
            fl = data_utils.DataLoader(fd, batch_size=1, shuffle=True, pin_memory=pin_memory)
            finetune_loader.append(fl)
            if mass_train_data is not None:
                mass_train_data[i].examples_list = []
        return finetune_loader, finetune_data

    @staticmethod
    def get_mass_loader(mass_train_paths, mt_model, num_processors, options, pin_memory):
        mass_train_data, mass_train_loader = [], []
        for i, mass_train_path in enumerate(mass_train_paths):
            td = dataset.MassDataset(batch_pickle_dir=mass_train_path,
                                     max_batch_capacity=num_processors * options.total_capacity,
                                     max_batch=num_processors * options.batch,
                                     pad_idx=mt_model.text_processor.pad_token_id(),
                                     max_seq_len=options.max_seq_len, keep_examples=True)
            mass_train_data.append(td)

            dl = data_utils.DataLoader(td, batch_size=1, shuffle=True, pin_memory=pin_memory)
            mass_train_loader.append(dl)
        return mass_train_data, mass_train_loader

    @staticmethod
    def get_img_loader(collator, dataset_class, img_train_loader, mt_model, num_batches, options, pin_memory,
                       transform, denom=1):
        if options.train_path is not None:
            img_train_loader = []
            train_paths = options.train_path.split(",")
            for train_path in train_paths:
                train_data = dataset_class(root_img_dir=options.image_dir,
                                           data_bin_file=train_path, transform=transform,
                                           max_capacity=int(options.img_capacity / denom),
                                           text_processor=mt_model.text_processor,
                                           max_img_per_batch=options.max_image)
                print(train_path, "Length of training data", len(train_data))
                tl = data_utils.DataLoader(train_data, batch_size=num_batches, shuffle=True,
                                           pin_memory=pin_memory,
                                           collate_fn=collator)
                img_train_loader.append(tl)

        return img_train_loader


if __name__ == "__main__":
    parser = get_img_options_parser()
    (options, args) = parser.parse_args()
    print(options)
    ImageDocTrainer.train(options=options)
    print("Finished Training!")
