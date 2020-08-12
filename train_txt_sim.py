import datetime
import os
import pickle
import sys
import time
from typing import List

import torch
import torch.utils.data as data_utils
from IPython.core import ultratb

from lm import LM
from option_parser import get_img_options_parser
from sen_sim import SenSim
from textprocessor import TextProcessor
from train_image_mt import ImageMTTrainer
from utils import build_optimizer, backward

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class SenSimTrainer(ImageMTTrainer):
    def train_epoch(self, step: int, saving_path: str = None,
                    mt_dev_iter: List[data_utils.DataLoader] = None,
                    mt_train_iter: List[data_utils.DataLoader] = None, max_step: int = 300000,
                    **kwargs):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0

        batch_zip, shortest = self.get_batch_zip(None, None, mt_train_iter)

        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        for i, batches in enumerate(batch_zip):
            for batch in batches:
                self.optimizer.zero_grad()
                try:
                    src_inputs = batch["src_texts"].squeeze(0)
                    src_mask = batch["src_pad_mask"].squeeze(0)
                    tgt_inputs = batch["dst_texts"].squeeze(0)
                    tgt_mask = batch["dst_pad_mask"].squeeze(0)
                    src_langs = batch["src_langs"].squeeze(0)
                    dst_langs = batch["dst_langs"].squeeze(0)
                    if src_inputs.size(0) < self.num_gpu:
                        continue
                    loss = self.model(src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                      src_mask=src_mask, tgt_mask=tgt_mask, src_langs=src_langs,
                                      tgt_langs=dst_langs, normalize=True)
                    nSens = src_inputs.size(0)

                    backward(loss, self.optimizer, self.fp16)

                    loss = float(loss.data) * nSens
                    tokens += nSens
                    total_tokens += nSens
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
                            if mt_dev_iter is not None and step % 5000 == 0:
                                dev_loss = self.eval(mt_dev_iter, saving_path)
                                print("Dev Loss:", dev_loss)

                            model.save(saving_path + ".latest")
                            with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
                                pickle.dump(self.optimizer, fp)

                        start, tokens, cur_loss = time.time(), 0, 0

                except RuntimeError as err:
                    print(repr(err))
                    torch.cuda.empty_cache()

            if i == shortest - 1:
                break
            if step >= max_step:
                break

        try:
            print("Total loss in this epoch: %f" % (total_loss / total_tokens))
            model.save(saving_path + ".latest")

            if mt_dev_iter is not None:
                dev_loss = self.eval(mt_dev_iter, saving_path)
                print("Dev Loss:", dev_loss)
        except RuntimeError as err:
            print(repr(err))

        return step

    def eval(self, dev_data_iter, saving_path):
        mt_output = []
        src_text = []
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()
        total_loss, total_items = 0, 0
        with torch.no_grad():
            for iter in dev_data_iter:
                for batch in iter:
                    src_inputs = batch["src_texts"].squeeze(0)
                    src_mask = batch["src_pad_mask"].squeeze(0)
                    tgt_inputs = batch["dst_texts"].squeeze(0)
                    src_langs = batch["src_langs"].squeeze(0)
                    dst_langs = batch["dst_langs"].squeeze(0)
                    tgt_mask = batch["dst_pad_mask"].squeeze(0)

                    loss = self.model(src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                      src_mask=src_mask, tgt_mask=tgt_mask, src_langs=src_langs,
                                      tgt_langs=dst_langs, normalize=True)
                    nSens = len(src_inputs)
                    total_loss += float(loss.data) * nSens
                    total_items += nSens

        model.train()
        total_loss /= total_items

        if total_loss <= self.best_loss:
            self.best_loss = total_loss
            print("Saving best Loss", self.best_loss)
            model.save(saving_path)
            with open(os.path.join(saving_path, "optim"), "wb") as fp:
                pickle.dump(self.optimizer, fp)

        return total_loss

    @staticmethod
    def train(options):
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)
        assert text_processor.pad_token_id() == 0
        num_processors = max(torch.cuda.device_count(), 1)

        mt_model = SenSim(text_processor=text_processor, enc_layer=options.encoder_layer, embed_dim=options.embed_dim,
                          intermediate_dim=options.intermediate_layer_dim)

        if options.lm_path is not None:
            lm = LM(text_processor=text_processor, enc_layer=options.encoder_layer,
                    embed_dim=options.embed_dim, intermediate_dim=options.intermediate_layer_dim)
            mt_model.init_from_lm(lm)

        print("Model initialization done!")

        optimizer = build_optimizer(mt_model, options.learning_rate, warump_steps=options.warmup)
        trainer = SenSimTrainer(model=mt_model, mask_prob=options.mask_prob, optimizer=optimizer, clip=options.clip,
                                fp16=options.fp16)

        pin_memory = torch.cuda.is_available()

        mt_train_loader = SenSimTrainer.get_mt_train_data(mt_model, num_processors, options, pin_memory)

        mt_dev_loader = None
        if options.mt_dev_path is not None:
            mt_dev_loader = SenSimTrainer.get_mt_dev_data(mt_model, options, pin_memory, text_processor, trainer, )

        step, train_epoch = 0, 1
        trainer.best_loss = 1000000
        while options.step > 0 and step < options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(mt_train_iter=mt_train_loader, max_step=options.step, mt_dev_iter=mt_dev_loader,
                                       saving_path=options.model_path, step=step)
            train_epoch += 1


if __name__ == "__main__":
    parser = get_img_options_parser()
    (options, args) = parser.parse_args()
    print(options)
    SenSimTrainer.train(options=options)
    print("Finished Training!")
