import datetime
import os
import pickle
import sys
import time
from optparse import OptionParser

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import transformers.optimization as optim
from IPython.core import ultratb

import dataset
from lm import LM
from parallel import DataParallelModel, DataParallelCriterion
from pytorch_lamb.pytorch_lamb import Lamb
from textprocessor import TextProcessor

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class LMTrainer:
    def __init__(self, model, mask_prob: float = 0.15, clip: int = 1, optimizer=None, warmup: int = 12500,
                 step: int = 125000, last_epoch: int = 0):
        self.model = model
        self.clip = clip
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if self.optimizer is not None:
            self.scheduler = optim.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup,
                                                                   num_training_steps=step + last_epoch)
            self.scheduler.last_epoch = last_epoch

        self.mask_prob = mask_prob
        self.criterion = nn.NLLLoss(ignore_index=model.text_processor.pad_token_id())

        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            print("Let's use", num_gpu, "GPUs!")
            self.model = DataParallelModel(self.model)
            self.criterion = DataParallelCriterion(self.criterion)

        self.best_dev_loss = float("inf")
        self.best_train_loss = float("inf")
        self.last_train_loss = float("inf")

    @staticmethod
    def build_optimizer(model, learning_rate, weight_decay):
        return Lamb(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(.9, .999), adam=True)

    def train_epoch(self, data_iter: data_utils.DataLoader, dev_data_iter: data_utils.DataLoader,
                    saving_path: str, step: int, max_grad_norm: float = 1.0):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0

        for i, batch in enumerate(data_iter):
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            model_to_call = self.model.module if hasattr(self.model, "module") else self.model
            mask, target, texts = LM.mask_text(self.mask_prob, batch["pad_mask"], batch["texts"],
                                               model_to_call.text_processor)
            try:
                predictions = self.model(device=self.device, mask=mask, texts=texts, pads=batch["pad_mask"],
                                         langs=batch["langs"])
                ntokens = target.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue

                loss = self.criterion(predictions, target).mean()
                loss.backward()

                LM.unmask_text(mask, target, texts)

                if self.optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    step += 1

                loss = float(loss.data) * ntokens
                total_loss += loss
                cur_loss += loss
                total_tokens += ntokens
                tokens += ntokens

                if step % 50 == 0:
                    elapsed = time.time() - start
                    print(datetime.datetime.now(),
                          "Epoch Step: %d Loss: %f Tokens per Sec: %f" % (step, cur_loss / tokens, tokens / elapsed))

                    if step % 500 == 0:
                        self.validate_and_save(saving_path, dev_data_iter)

                    start, tokens, cur_loss = time.time(), 0, 0
            except RuntimeError as err:
                print("Problem with batch item", texts.size())
                torch.cuda.empty_cache()
                pass

        current_loss = total_loss / total_tokens
        print("Total loss in this epoch: %f" % current_loss)
        if current_loss < self.best_train_loss:
            self.best_train_loss = current_loss
            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            model_to_save.save(saving_path + ".latest")
            with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
                pickle.dump((self.optimizer, self.scheduler.last_epoch), fp)
        self.last_train_loss = current_loss

        self.validate_and_save(saving_path, dev_data_iter)
        return step

    def validate_and_save(self, saving_path, dev_data_iter):
        with torch.no_grad():
            model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            model.eval()
            total_dev_loss, total_dev_tokens = 0, 0
            for batch in dev_data_iter:
                model_to_call = self.model.module if hasattr(self.model, "module") else self.model
                mask, target, texts = LM.mask_text(self.mask_prob, batch["pad_mask"], batch["texts"].clone(),
                                                   model_to_call.text_processor)
                predictions = self.model(device=self.device, mask=mask, texts=texts, pads=batch["pad_mask"],
                                         langs=batch["langs"])
                ntokens = target.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue
                loss = self.criterion(predictions, target).mean().data * ntokens
                total_dev_loss += float(loss)
                total_dev_tokens += ntokens

            dev_loss = total_dev_loss / total_dev_tokens
            print("Current dev loss", dev_loss)
            if self.best_dev_loss > float(dev_loss):
                self.best_dev_loss = float(dev_loss)
                print("saving best dev loss", self.best_dev_loss)
                model_to_save = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )
                model_to_save.save(saving_path)
                with open(os.path.join(saving_path, "optim"), "wb") as fp:
                    pickle.dump((self.optimizer, self.scheduler.last_epoch), fp)
            model.train()

    @staticmethod
    def config_dropout(model, dropout):
        model.encoder.config.hidden_dropout_prob = dropout
        model.encoder.config.attention_probs_dropout_prob = dropout

    @staticmethod
    def train(options):
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)

        if options.pretrained_path is None:
            lm = LM(text_processor=text_processor, size=options.model_size)
        else:
            lm = LM.load(options.pretrained_path)
        LMTrainer.config_dropout(lm, options.dropout)

        train_data = dataset.TextDataset(save_cache_dir=options.train_path, max_cache_size=options.cache_size)
        dev_data = dataset.TextDataset(save_cache_dir=options.dev_path, max_cache_size=options.cache_size,
                                       load_all=True)

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer, last_epoch = pickle.load(fp)
        else:
            optimizer, last_epoch = LMTrainer.build_optimizer(lm, options.learning_rate, options.weight_decay), 0

        trainer = LMTrainer(model=lm, mask_prob=options.mask_prob, optimizer=optimizer, clip=options.clip,
                            warmup=options.warmup, step=options.step, last_epoch=last_epoch)

        collator = dataset.TextCollator(pad_idx=text_processor.pad_token_id())
        train_sampler, dev_sampler = None, None

        pin_memory = torch.cuda.is_available()
        loader = data_utils.DataLoader(train_data, batch_size=options.batch, shuffle=False, pin_memory=pin_memory,
                                       collate_fn=collator, sampler=train_sampler)
        dev_loader = data_utils.DataLoader(dev_data, batch_size=options.batch, shuffle=False, pin_memory=pin_memory,
                                           collate_fn=collator, sampler=dev_sampler)

        step, train_epoch = 0, 1
        while step <= options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(data_iter=loader, dev_data_iter=dev_loader, saving_path=options.model_path,
                                       step=step)


def get_option_parser():
    parser = OptionParser()
    parser.add_option("--train", dest="train_path", help="Path to the train data pickle files for large data",
                      metavar="FILE", default=None)
    parser.add_option("--dev", dest="dev_path", help="Path to the dev data pickle files for large data", metavar="FILE",
                      default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--cache_size", dest="cache_size", help="Number of blocks in cache", type="int", default=300)
    parser.add_option("--model", dest="model_path", help="Directory path to save the best model", metavar="FILE",
                      default=None)
    parser.add_option("--pretrained", dest="pretrained_path", help="Directory of pretrained model", metavar="FILE",
                      default=None)
    parser.add_option("--epoch", dest="num_epochs", help="Number of training epochs", type="int", default=100)
    parser.add_option("--clip", dest="clip", help="For gradient clipping", type="int", default=1)
    parser.add_option("--batch", dest="batch", help="Batch size", type="int", default=512)
    parser.add_option("--mask", dest="mask_prob", help="Random masking probability", type="float", default=0.15)
    parser.add_option("--embed", dest="d_model", help="Embedding of contextual word vectors", type="int", default=768)
    parser.add_option("--lr", dest="learning_rate", help="Learning rate", type="float", default=0.0025)
    parser.add_option("--warmup", dest="warmup", help="Number of warmup steps", type="int", default=12500)
    parser.add_option("--step", dest="step", help="Number of training steps", type="int", default=125000)
    parser.add_option("--decay", dest="weight_decay", help="Weight decay", type="float", default=0.01)
    parser.add_option("--max_grad_norm", dest="max_grad_norm", help="Max grad norm", type="float", default=1.0)
    parser.add_option("--cont", action="store_true", dest="continue_train",
                      help="Continue training from pretrained model", default=False)
    parser.add_option("--dropout", dest="dropout", help="Dropout probability", type="float", default=0.1)
    parser.add_option("--dff", dest="d_ff", help="Position-wise feed-forward dimensions", type="int", default=2048)
    parser.add_option("--size", dest="model_size", help="Model size: 3 (base), 2 (medium), 1 (small)", type="int",
                      default=1)
    return parser


if __name__ == "__main__":
    parser = get_option_parser()
    (options, args) = parser.parse_args()
    print(options)
    LMTrainer.train(options=options)
    print("Finished Training!")
