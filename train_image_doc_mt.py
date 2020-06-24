import copy
import datetime
import os
import pickle
import sys
import time
from typing import List

import torch
import torch.utils.data as data_utils
import transformers.optimization as optim
from IPython.core import ultratb
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

import dataset
from albert_seq2seq import MassSeq2Seq
from image_doc_model import ImageSeq2Seq
from lm import LM
from option_parser import get_img_options_parser
from parallel import DataParallelModel
from seq_gen import get_outputs_until_eos
from textprocessor import TextProcessor
from train_mass import MassTrainer
from utils import build_optimizer, mass_mask, mass_unmask

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class ImageDocTrainer(MassTrainer):
    def __init__(self, model, mask_prob: float = 0.3, clip: int = 1, optimizer=None, warmup: int = 12500,
                 step: int = 125000, beam_width: int = 5, max_len_a: float = 1.1, max_len_b: int = 5,
                 len_penalty_ratio: float = 0.8, self_translate: bool = False, last_epoch: int = 0,
                 nll_loss: bool = False):
        super().__init__(model, mask_prob, clip, optimizer, warmup, step, beam_width, max_len_a, max_len_b,
                         len_penalty_ratio, self_translate, last_epoch, nll_loss)

        self.mass_model = MassSeq2Seq(config=model.config, encoder=model.encoder, decoder=model.decoder,
                                      output_layer=model.output_layer,
                                      text_processor=model.text_processor, checkpoint=model.checkpoint)
        if self.num_gpu > 1:
            self.mass_model = DataParallelModel(self.mass_model)

    def train_epoch(self, data_iter: data_utils.DataLoader, mass_data_iter: List[data_utils.DataLoader], step: int,
                    saving_path: str = None, mt_dev_iter: data_utils.DataLoader = None, fine_tune: bool = False,
                    lang_directions: dict = False, **kwargs):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        sentences = 0
        shortest = min([len(l) for l in mass_data_iter] + [len(data_iter)])
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        for i, batches in enumerate(zip(data_iter, mass_data_iter[0], mass_data_iter[1])):
            for batch in batches:
                self.optimizer.zero_grad()
                is_img_batch = isinstance(batch, list) and "captions" in batch[0]
                try:
                    if is_img_batch:  # Image data
                        if len(batch) < self.num_gpu:
                            continue
                        predictions = self.model(device=self.device, batch=batch, log_softmax=True)
                        targets = [b["captions"][:, 1:].contiguous().view(-1) for b in batch]
                        tgt_mask_flat = [b["caption_mask"][:, 1:].contiguous().view(-1) for b in batch]
                        targets = torch.cat([targets[i][tgt_mask_flat[i]] for i in range(len(batch))])
                        ntokens = targets.size(0)
                        sentences += sum([int(b["docs"].size(0)) + int(b["captions"].size(0)) for b in batch])
                    else:  # MASS data
                        src_inputs = batch["src_texts"].squeeze(0)
                        src_pad_mask = batch["src_pad_mask"].squeeze(0)
                        pad_indices = batch["pad_idx"].squeeze(0)
                        if src_inputs.size(0) < self.num_gpu:
                            continue

                        if not fine_tune:
                            masked_info = mass_mask(self.mask_prob, pad_indices, src_inputs, model.text_processor)
                            predictions = self.mass_model(device=self.device, src_inputs=masked_info["src_text"],
                                                          tgt_inputs=masked_info["to_recover"],
                                                          tgt_positions=masked_info["positions"], src_pads=src_pad_mask,
                                                          pad_idx=model.text_processor.pad_token_id(),
                                                          src_langs=batch["langs"].squeeze(0),
                                                          log_softmax=True)
                            targets = masked_info["targets"]
                            ntokens = targets.size(0)
                        else:
                            target_langs = torch.LongTensor([lang_directions[int(l)] for l in src_inputs[:, 0]])
                            dst_langs = torch.LongTensor(
                                [model.text_processor.languages[model.text_processor.id2token(lang_directions[int(l)])]
                                 for l in src_inputs[:, 0]])

                            model.eval()
                            with torch.no_grad():
                                # We do not backpropagate the data generator following the MASS paper.
                                outputs = self.generator(device=self.device, src_inputs=src_inputs,
                                                         src_sizes=pad_indices,
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
                            predictions = self.mass_model(device=self.device, src_inputs=translations,
                                                          tgt_inputs=src_inputs,
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

                    loss = self.criterion(predictions, targets).mean()
                    loss.backward()

                    loss = float(loss.data) * ntokens
                    total_loss += loss
                    cur_loss += loss
                    total_tokens += ntokens
                    tokens += ntokens

                    # We accumulate the gradients for both tasks!
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    step += 1

                    if not is_img_batch and not fine_tune:
                        mass_unmask(masked_info["src_text"], masked_info["src_mask"], masked_info["mask_idx"])

                except RuntimeError as err:
                    print(err)
                    print("Error processing", is_img_batch)

                if step % 50 == 0 and tokens > 0:
                    elapsed = time.time() - start
                    print(datetime.datetime.now(),
                          "Epoch Step: %d Loss: %f Tokens per Sec: %f Sentences per Sec: %f" % (
                              step, cur_loss / tokens, tokens / elapsed, sentences / elapsed))

                    if step % 500 == 0:
                        if mt_dev_iter is not None:
                            bleu = self.eval_bleu(mt_dev_iter, saving_path)
                            print("Pretraining BLEU:", bleu)

                        model.save(saving_path + ".latest")
                        with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
                            pickle.dump(
                                (self.optimizer, self.scheduler.last_epoch if self.scheduler is not None else step), fp)

                    start, tokens, cur_loss, sentences = time.time(), 0, 0, 0
            if i == shortest - 1:
                break

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        model.save(saving_path + ".latest")

        if mt_dev_iter is not None:
            bleu = self.eval_bleu(mt_dev_iter, saving_path)
            print("Pretraining BLEU:", bleu)

        return step

    @staticmethod
    def train(options):
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)

        if options.pretrained_path is not None:
            mt_model, lm = ImageSeq2Seq.load(options.pretrained_path)
        else:
            if options.lm_path is None:
                lm = LM(text_processor=text_processor, size=options.model_size)
            else:
                lm = LM.load(options.lm_path)

            decoder = copy.deepcopy(lm.encoder) if options.sep_encoder else lm.encoder
            mt_model = ImageSeq2Seq(config=lm.config, encoder=lm.encoder, decoder=decoder, output_layer=lm.masked_lm,
                                    text_processor=lm.text_processor, checkpoint=options.checkpoint)

        transform = transforms.Compose([  # [1]
            transforms.Resize(256),  # [2]
            transforms.CenterCrop(224),  # [3]
            transforms.ToTensor(),  # [4]
            transforms.Normalize(  # [5]
                mean=[0.485, 0.456, 0.406],  # [6]
                std=[0.229, 0.224, 0.225]  # [7]
            )])

        print("Model initialization done!")

        train_data = dataset.ImageDocDataset(root_img_dir=options.image_dir,
                                             data_bin_file=options.train_path, transform=transform,
                                             max_doc_batch_capacity=options.img_capacity,
                                             text_processor=mt_model.text_processor,
                                             max_img_per_batch=options.max_image)

        pin_memory = torch.cuda.is_available()

        # We assume that the collator function returns a list with the size of number of gpus (in case of cpus,
        collator = dataset.ImageTextCollator()
        num_batches = max(1, torch.cuda.device_count())

        train_loader = data_utils.DataLoader(train_data, batch_size=num_batches, shuffle=False, pin_memory=pin_memory,
                                             collate_fn=collator)

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer, last_epoch = pickle.load(fp)
        else:
            optimizer, last_epoch = build_optimizer(mt_model, options.learning_rate, options.weight_decay,
                                                    use_adam=options.adam), 0
        trainer = ImageDocTrainer(model=mt_model, mask_prob=options.mask_prob, optimizer=optimizer, clip=options.clip,
                                  warmup=options.warmup, step=options.step,
                                  beam_width=options.beam_width, max_len_a=options.max_len_a,
                                  max_len_b=options.max_len_b, len_penalty_ratio=options.len_penalty_ratio,
                                  last_epoch=last_epoch)

        mass_train_data, mass_train_loader, finetune_loader, mt_dev_loader = None, None, None, None
        mass_train_paths = options.mass_train_path.strip().split(",")
        if options.step > 0 and last_epoch < options.step:
            mass_train_data, mass_train_loader = [], []
            for i, mass_train_path in enumerate(mass_train_paths):
                td = dataset.MassDataset(batch_pickle_dir=mass_train_path,
                                         max_batch_capacity=options.total_capacity, max_batch=options.batch,
                                         pad_idx=mt_model.text_processor.pad_token_id(),
                                         max_seq_len=options.max_seq_len, keep_examples=True)
                mass_train_data.append(td)
                dl = data_utils.DataLoader(td, batch_size=1, shuffle=True, pin_memory=pin_memory)
                mass_train_loader.append(dl)

        lang_directions = {}
        if options.finetune_step > 0:
            finetune_data, finetune_loader = [], []
            for i, mass_train_path in enumerate(mass_train_paths):
                fd = dataset.MassDataset(batch_pickle_dir=mass_train_path,
                                         max_batch_capacity=int(options.total_capacity / 2),
                                         max_batch=int(options.batch / 2),
                                         pad_idx=mt_model.text_processor.pad_token_id(),
                                         max_seq_len=options.max_seq_len, keep_examples=False,
                                         example_list=None if mass_train_data is None else mass_train_data[
                                             i].examples_list)
                finetune_data.append(fd)
                fl = data_utils.DataLoader(fd, batch_size=1, shuffle=True, pin_memory=pin_memory)
                finetune_loader.append(fl)

                if mass_train_data is not None:
                    mass_train_data[i].examples_list = []

            langs = set()
            for fd in finetune_data:
                for lang1 in fd.lang_ids:
                    langs.add(lang1)

            for lang1 in langs:
                for lang2 in langs:
                    if lang1 != lang2:
                        # Assuming that we only have two languages!
                        lang_directions[lang1] = lang2

        mt_dev_loader = None
        if options.mt_dev_path is not None:
            mt_dev_data = dataset.MTDataset(batch_pickle_dir=options.mt_dev_path,
                                            max_batch_capacity=options.total_capacity,
                                            max_batch=int(options.batch / (options.beam_width * 2)),
                                            pad_idx=mt_model.text_processor.pad_token_id())
            mt_dev_loader = data_utils.DataLoader(mt_dev_data, batch_size=1, shuffle=False, pin_memory=pin_memory)

            print("creating reference")
            trainer.reference = []

            generator = (
                trainer.generator.module if hasattr(trainer.generator, "module") else trainer.generator
            )

            for batch in mt_dev_loader:
                tgt_inputs = batch["dst_texts"].squeeze()
                refs = get_outputs_until_eos(text_processor.sep_token_id(), tgt_inputs)
                ref = [generator.seq2seq_model.text_processor.tokenizer.decode(ref.numpy()) for ref in refs]
                trainer.reference += ref

        step, train_epoch = 0, 1
        while options.step > 0 and step < options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(data_iter=train_loader, mass_data_iter=mass_train_loader,
                                       mt_dev_iter=mt_dev_loader, saving_path=options.model_path, step=step)
            train_epoch += 1

        finetune_epoch = 0
        mt_model.save(options.model_path + ".beam")
        if train_epoch > 0:
            # Resetting the optimizer for the purpose of finetuning.
            model = mt_model.module if hasattr(mt_model, "module") else mt_model
            trainer.optimizer = build_optimizer(model, options.learning_rate, options.weight_decay,
                                                use_adam=options.adam)
            trainer.scheduler = optim.get_linear_schedule_with_warmup(trainer.optimizer,
                                                                      num_warmup_steps=options.warmup,
                                                                      num_training_steps=options.finetune_step)

        while options.finetune_step > 0 and step <= options.finetune_step + options.step:
            print("finetune epoch", finetune_epoch)
            step = trainer.train_epoch(data_iter=train_loader, mass_data_iter=finetune_loader,
                                       mt_dev_iter=mt_dev_loader, saving_path=options.model_path, step=step,
                                       fine_tune=True, lang_directions=lang_directions)
            finetune_epoch += 1


if __name__ == "__main__":
    parser = get_img_options_parser()
    (options, args) = parser.parse_args()
    print(options)
    ImageDocTrainer.train(options=options)
    print("Finished Training!")
