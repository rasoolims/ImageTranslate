import datetime
import os
import pickle
import sys
import time
from collections import defaultdict

import sacrebleu
import torch.utils.data as data_utils
from IPython.core import ultratb

import dataset
from faster_rcnn_feats import *
from image_model import ImageCaptioning, ImageMassSeq2Seq
from option_parser import get_img_options_parser
from seq2seq import Seq2Seq
from seq_gen import get_outputs_until_eos
from textprocessor import TextProcessor
from train_image_mt import ImageMTTrainer, get_lex_dict
from utils import build_optimizer, backward

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class ImageCaptionTrainer(ImageMTTrainer):
    def train_epoch(self, img_data_iter: List[data_utils.DataLoader], step: int, saving_path: str = None,
                    img_dev_data_iter: List[data_utils.DataLoader] = None, max_step: int = 300000,
                    lex_dict=None, accum=1, mt_train_iter: List[data_utils.DataLoader] = None,
                    mt_dev_iter: List[data_utils.DataLoader] = None, mtl_weight=0.1, **kwargs):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        batch_zip, shortest = self.get_batch_zip(img_data_iter, None, mt_train_iter)

        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        for i, batches in enumerate(batch_zip):
            for batch in batches:
                try:
                    is_img_batch = isinstance(batch, list) and "captions" in batch[0]
                    if is_img_batch:  # Captioning training data.
                        captions = [b["captions"] for b in batch]
                        caption_pad_mask = [b["caption_mask"] for b in batch]
                        proposals = [b["proposal"] for b in batch] if lex_dict is not None else None
                        langs = [b["langs"] for b in batch]
                        if len(batch) < self.num_gpu:
                            continue

                        predictions = self.model(tgt_inputs=captions,
                                                 tgt_mask=caption_pad_mask,
                                                 pad_idx=model.text_processor.pad_token_id(),
                                                 tgt_langs=langs, batch=batch, proposals=proposals,
                                                 log_softmax=True)
                        targets = torch.cat(list(map(lambda c: c[:, 1:].contiguous().view(-1), captions)))
                        tgt_mask_flat = torch.cat(list(map(lambda c: c[:, 1:].contiguous().view(-1), caption_pad_mask)))
                        targets = targets[tgt_mask_flat]
                    else:  # MT data!
                        src_inputs = batch["src_texts"].squeeze(0)
                        src_mask = batch["src_pad_mask"].squeeze(0)
                        tgt_inputs = batch["dst_texts"].squeeze(0)
                        tgt_mask = batch["dst_pad_mask"].squeeze(0)
                        src_langs = batch["src_langs"].squeeze(0)
                        dst_langs = batch["dst_langs"].squeeze(0)
                        proposals = batch["proposal"].squeeze(0) if lex_dict is not None else None
                        if src_inputs.size(0) < self.num_gpu:
                            continue
                        predictions = self.model(src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                                 src_pads=src_mask, tgt_mask=tgt_mask, src_langs=src_langs,
                                                 tgt_langs=dst_langs, proposals=proposals,
                                                 pad_idx=model.text_processor.pad_token_id(), log_softmax=True)
                        targets = tgt_inputs[:, 1:].contiguous().view(-1)
                        tgt_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
                        targets = targets[tgt_mask_flat]
                    ntokens = targets.size(0)

                    if ntokens > 0:
                        if self.num_gpu == 1:
                            targets = targets.to(predictions.device)

                        loss = self.criterion(predictions, targets).mean()
                        weight = 1 if is_img_batch else mtl_weight
                        backward(loss * weight, self.optimizer, self.fp16)

                        loss = float(loss.data) * ntokens
                        tokens += ntokens
                        total_tokens += ntokens
                        total_loss += loss
                        cur_loss += loss

                        # We accumulate the gradients for both tasks!
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                        step += 1

                        if step % accum == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                        if step % 50 == 0 and tokens > 0:
                            elapsed = time.time() - start
                            print(datetime.datetime.now(),
                                  "Epoch Step: %d Loss: %f Tokens per Sec: %f " % (
                                      step, cur_loss / tokens, tokens / elapsed))

                            if step % 500 == 0:
                                if img_dev_data_iter is not None and step % 5000 == 0:
                                    bleu = self.eval_bleu(img_dev_data_iter, saving_path)
                                    print("Captioning BLEU:", bleu)
                                if mt_dev_iter is not None and step % 5000 == 0:
                                    bleu = super().eval_bleu(mt_dev_iter, saving_path)
                                    print("MT BLEU:", bleu)

                                model.save(saving_path + ".latest")
                                with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
                                    pickle.dump(self.optimizer, fp)

                            start, tokens, cur_loss = time.time(), 0, 0

                        if step >= max_step:
                            break
                        if i == shortest - 1:
                            break
                except RuntimeError as err:
                    print(repr(err))
                    torch.cuda.empty_cache()

        try:
            if img_dev_data_iter is not None:
                bleu = self.eval_bleu(img_dev_data_iter, saving_path)
                print("Captioning BLEU:", bleu)
            if mt_dev_iter is not None:
                bleu = super().eval_bleu(mt_dev_iter, saving_path)
                print("MT BLEU:", bleu)

            print("Total loss in this epoch: %f" % (total_loss / total_tokens))
            model.save(saving_path + ".latest")
        except RuntimeError as err:
            print(repr(err))

        return step

    def eval_bleu(self, dev_data_iter, saving_path):
        mt_output = []
        mt_ids = []
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()

        with torch.no_grad():
            for iter in dev_data_iter:
                for batch in iter:
                    proposals = [b["proposal"] for b in batch]
                    dst_langs = [b["langs"] for b in batch]
                    img_ids = [b["img_ids"] for b in batch][0]
                    first_tokens = [b["first_tokens"] for b in batch]
                    images = [b["images"] for b in batch]
                    max_len = [b["max_len"] for b in batch][0]
                    outputs = self.generator(images=images, first_tokens=first_tokens, tgt_langs=dst_langs,
                                             pad_idx=model.text_processor.pad_token_id(), proposals=proposals,
                                             max_len=max_len)
                    if self.num_gpu > 1:
                        new_outputs = []
                        for output in outputs:
                            new_outputs += output
                        outputs = new_outputs

                    mt_output += list(map(lambda x: model.text_processor.tokenizer.decode(x[1:].numpy()), outputs))
                    mt_ids += img_ids
            model.train()
        references = list(map(lambda id: self.caption_reference[id], mt_ids))
        max_reflen = max(map(lambda x: len(x), references))
        all_refs = list(map(lambda l: list(map(lambda r: r[l] if l < len(r) else None, references)), range(max_reflen)))
        bleu = sacrebleu.corpus_bleu(mt_output, all_refs, lowercase=True, tokenize="intl")

        output = "\n".join(["\nOutput:\n" + o + "\n\nReferences:\n" + "\n".join(
            self.caption_reference[mt_ids[i]]) + "\n\n***************\n" for i, o in enumerate(mt_output)])
        with open(os.path.join(saving_path, "bleu.caption.output"), "w") as writer:
            writer.write(output)

        if bleu.score > self.best_bleu:
            self.best_bleu = bleu.score
            print("Saving best BLEU", self.best_bleu)
            model.save(saving_path)
            with open(os.path.join(saving_path, "optim"), "wb") as fp:
                pickle.dump(self.optimizer, fp)

            with open(os.path.join(saving_path, "bleu.caption.best.output"), "w") as writer:
                writer.write(output)

        return bleu.score

    @staticmethod
    def train(options):
        lex_dict = None
        if options.dict_path is not None:
            lex_dict = get_lex_dict(options.dict_path)
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)
        assert text_processor.pad_token_id() == 0

        if options.pretrained_path is not None:
            caption_model = Seq2Seq.load(ImageCaptioning, options.pretrained_path, tok_dir=options.tokenizer_path)
        else:
            caption_model = ImageCaptioning(use_proposals=lex_dict is not None, tie_embed=options.tie_embed,
                                            text_processor=text_processor, resnet_depth=options.resnet_depth,
                                            lang_dec=options.lang_decoder, enc_layer=options.encoder_layer,
                                            dec_layer=options.decoder_layer, embed_dim=options.embed_dim,
                                            intermediate_dim=options.intermediate_layer_dim, use_obj=not options.no_obj,num_img_encode_layers=options.img_enc)

        if options.lm_path is not None:  # In our case, this is an MT model.
            mt_pret_model = Seq2Seq.load(ImageMassSeq2Seq, options.lm_path, tok_dir=options.tokenizer_path)
            caption_model.encoder = mt_pret_model.encoder
            caption_model.decoder = mt_pret_model.decoder
            caption_model.output_layer = mt_pret_model.output_layer

        print("Model initialization done!")

        # We assume that the collator function returns a list with the size of number of gpus (in case of cpus,
        collator = dataset.ImageTextCollator()
        num_batches = max(1, torch.cuda.device_count())

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer = pickle.load(fp)
        else:
            optimizer = build_optimizer(caption_model, options.learning_rate, warump_steps=options.warmup)
        trainer = ImageCaptionTrainer(model=caption_model, mask_prob=options.mask_prob, optimizer=optimizer,
                                      clip=options.clip,
                                      beam_width=options.beam_width, max_len_a=options.max_len_a,
                                      max_len_b=options.max_len_b, len_penalty_ratio=options.len_penalty_ratio,
                                      fp16=options.fp16, mm_mode=options.mm_mode)

        pin_memory = torch.cuda.is_available()
        img_train_loader = ImageMTTrainer.get_img_loader(collator, dataset.ImageCaptionDataset, options.train_path,
                                                         caption_model, num_batches, options, pin_memory,
                                                         lex_dict=lex_dict, shuffle=(options.local_rank < 0))
        num_processors = max(torch.cuda.device_count(), 1) if options.local_rank < 0 else 1
        mt_train_loader = None
        if options.mt_train_path is not None:
            mt_train_loader = ImageMTTrainer.get_mt_train_data(caption_model, num_processors, options, pin_memory,
                                                               lex_dict=lex_dict)

        img_dev_loader = ImageMTTrainer.get_img_loader(collator, dataset.ImageCaptionTestDataset, options.dev_path,
                                                       caption_model, num_batches, options, pin_memory,
                                                       lex_dict=lex_dict,
                                                       shuffle=False, denom=2)

        trainer.caption_reference = None
        if img_dev_loader is not None:
            trainer.caption_reference = defaultdict(list)
            generator = (
                trainer.generator.module if hasattr(trainer.generator, "module") else trainer.generator
            )
            for data in img_dev_loader:
                for batch in data:
                    for b in batch:
                        captions = b["captions"]
                        for id in captions:
                            for caption in captions[id]:
                                refs = get_outputs_until_eos(text_processor.sep_token_id(), caption,
                                                             remove_first_token=True)
                                ref = [generator.seq2seq_model.text_processor.tokenizer.decode(ref.numpy()) for ref in
                                       refs]
                                trainer.caption_reference[id] += ref
            print("Number of dev image/captions", len(trainer.caption_reference))

        mt_dev_loader = None
        if options.mt_dev_path is not None:
            mt_dev_loader = ImageMTTrainer.get_mt_dev_data(caption_model, options, pin_memory, text_processor, trainer,
                                                           lex_dict=lex_dict)
            print("Number of dev sentences", len(trainer.reference))

        step, train_epoch = 0, 1
        while options.step > 0 and step < options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(img_data_iter=img_train_loader, img_dev_data_iter=img_dev_loader,
                                       max_step=options.step, lex_dict=lex_dict, mt_train_iter=mt_train_loader,
                                       saving_path=options.model_path, step=step, accum=options.accum,
                                       mt_dev_iter=mt_dev_loader, mtl_weight=options.mtl_weight)
            train_epoch += 1


if __name__ == "__main__":
    parser = get_img_options_parser()
    (options, args) = parser.parse_args()
    print(options)
    ImageCaptionTrainer.train(options=options)
    print("Finished Training!")
