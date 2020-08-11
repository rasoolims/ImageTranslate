import datetime
import os
import pickle
import sys
import time
from typing import List

import torch
import torch.utils.data as data_utils
from IPython.core import ultratb
from apex import amp

import dataset
from image_model import ImageCaptioning, Caption2Image
from option_parser import get_img_options_parser
from parallel import DataParallelModel
from seq2seq import Seq2Seq
from textprocessor import TextProcessor
from train_image_mt import ImageMTTrainer, get_lex_dict
from utils import build_optimizer, backward

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class Caption2ImageTrainer(ImageMTTrainer):
    def __init__(self, model, caption_model, mask_prob: float = 0.3, clip: int = 1, optimizer=None,
                 beam_width: int = 5, max_len_a: float = 1.1, max_len_b: int = 5,
                 len_penalty_ratio: float = 0.8, nll_loss: bool = False, fp16: bool = False, mm_mode="mixed"):
        super().__init__(model, mask_prob, clip, optimizer, beam_width, max_len_a, max_len_b, len_penalty_ratio,
                         nll_loss, fp16, mm_mode)
        self.caption_model = caption_model
        self.caption_model = self.caption_model.to(self.device)

        if self.num_gpu == 1 and fp16:
            self.caption_model = amp.initialize(self.caption_model, opt_level="O2")

        if self.num_gpu > 1:
            print("Let's use", self.num_gpu, "GPUs!")
            self.caption_model = DataParallelModel(self.caption_model)

    def train_epoch(self, img_data_iter: List[data_utils.DataLoader], step: int, saving_path: str = None,
                    img_dev_data_iter: List[data_utils.DataLoader] = None, max_step: int = 300000,
                    lex_dict=None, **kwargs):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        batch_zip, shortest = self.get_batch_zip(img_data_iter, None, None)

        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        for i, batches in enumerate(batch_zip):
            for batch in batches:
                try:
                    self.optimizer.zero_grad()
                    captions = [b["captions"] for b in batch]
                    caption_pad_mask = [b["caption_mask"] for b in batch]
                    langs = [b["langs"] for b in batch]

                    with torch.no_grad():
                        image_encoding = self.caption_model(batch=batch, encode_only=True)
                        image_encoding = image_encoding.view(image_encoding.size(0), -1)

                    predictions = self.model(src_inputs=captions, src_mask=caption_pad_mask, src_langs=langs)
                    l2_loss = torch.dist(predictions, image_encoding, 2) / predictions.size(0)
                    backward(l2_loss, self.optimizer, self.fp16)

                    loss = float(l2_loss.data)
                    tokens += int(predictions.size(0))
                    total_tokens += int(predictions.size(0))
                    total_loss += loss
                    cur_loss += loss

                    # We accumulate the gradients for both tasks!
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    step += 1

                    if step % 50 == 0 and tokens > 0:
                        elapsed = time.time() - start
                        print(datetime.datetime.now(),
                              "Epoch Step: %d Loss: %f Tokens per Sec: %f " % (
                                  step, cur_loss / tokens, tokens / elapsed))

                        if step % 500 == 0:
                            if img_dev_data_iter is not None and step % 5000 == 0:
                                loss = self.eval_bleu(img_dev_data_iter)
                                print("Dev Loss:", loss)

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
            print("Total loss in this epoch: %f" % (total_loss / total_tokens))
            model.save(saving_path + ".latest")

            loss = self.eval(img_dev_data_iter)
            print("Dev Loss:", loss)
        except RuntimeError as err:
            print(repr(err))

        return step

    def eval(self, img_dev_data_iter: List[data_utils.DataLoader]):
        total_loss, tokens, = 0, 0

        for data in img_dev_data_iter:
            for batch in data:
                try:
                    with torch.no_grad():
                        captions = [b["captions"] for b in batch]
                        caption_pad_mask = [b["caption_mask"] for b in batch]
                        langs = [b["langs"] for b in batch]

                        image_encoding = self.caption_model(batch=batch, encode_only=True)
                        image_encoding = image_encoding.view(image_encoding.size(0), -1)

                        predictions = self.model(src_inputs=captions, src_mask=caption_pad_mask, src_langs=langs)
                        l2_loss = torch.dist(predictions, image_encoding, 2) / predictions.size(0)

                        loss = float(l2_loss.data)
                        tokens += int(predictions.size(0))
                        total_loss += loss
                except RuntimeError as err:
                    print(repr(err))
                    pass

        return total_loss / tokens

    @staticmethod
    def train(options):
        lex_dict = None
        if options.dict_path is not None:
            lex_dict = get_lex_dict(options.dict_path)
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)
        assert text_processor.pad_token_id() == 0

        image_captioner = Seq2Seq.load(ImageCaptioning, options.pretrained_path, tok_dir=options.tokenizer_path)
        txt2ImageModel = Caption2Image(text_processor=text_processor, enc_layer=options.encoder_layer,
                                       embed_dim=options.embed_dim, intermediate_dim=options.intermediate_layer_dim)

        print("Model initialization done!")

        # We assume that the collator function returns a list with the size of number of gpus (in case of cpus,
        collator = dataset.ImageTextCollator()
        num_batches = max(1, torch.cuda.device_count())

        optimizer = build_optimizer(txt2ImageModel, options.learning_rate, warump_steps=options.warmup)

        trainer = Caption2ImageTrainer(model=txt2ImageModel, caption_model=image_captioner, mask_prob=options.mask_prob,
                                       optimizer=optimizer,
                                       clip=options.clip,
                                       beam_width=options.beam_width, max_len_a=options.max_len_a,
                                       max_len_b=options.max_len_b, len_penalty_ratio=options.len_penalty_ratio,
                                       fp16=options.fp16, mm_mode=options.mm_mode)

        pin_memory = torch.cuda.is_available()
        img_train_loader = ImageMTTrainer.get_img_loader(collator, dataset.ImageCaptionDataset, options.train_path,
                                                         txt2ImageModel, num_batches, options, pin_memory,
                                                         lex_dict=lex_dict)

        img_dev_loader = ImageMTTrainer.get_img_loader(collator, dataset.ImageCaptionDataset, options.dev_path,
                                                       txt2ImageModel, num_batches, options, pin_memory,
                                                       lex_dict=lex_dict,
                                                       shuffle=False, denom=2)

        step, train_epoch = 0, 1
        while options.step > 0 and step < options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(img_data_iter=img_train_loader, img_dev_data_iter=img_dev_loader,
                                       max_step=options.step, lex_dict=lex_dict,
                                       saving_path=options.model_path, step=step)
            train_epoch += 1


if __name__ == "__main__":
    parser = get_img_options_parser()
    (options, args) = parser.parse_args()
    print(options)
    Caption2ImageTrainer.train(options=options)
    print("Finished Training!")
