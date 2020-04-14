import datetime
import os
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


class Trainer:
    def __init__(self, model: LM, mask_prob: float = 0.15, clip: int = 1, optimizer=None, warmup: float = 0.1,
                 warmup_steps: int = 125000):
        self.model = model

        self.clip = clip
        self.optimizer = optimizer
        if optimizer is not None:
            self.scheduler = optim.get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=int(warmup * warmup_steps), num_training_steps=warmup_steps
            )
        self.mask_prob = mask_prob
        self.criterion = nn.NLLLoss(ignore_index=model.text_processor.pad_token_id())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            print("Let's use", num_gpu, "GPUs!")
            self.model = DataParallelModel(self.model)
            self.criterion = DataParallelCriterion(self.criterion)
        self.model = self.model.to(self.device)

    @staticmethod
    def build_optimizer(model, learning_rate, weight_decay):
        return Lamb(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(.9, .999), adam=True)

    def train_epoch(self, data_iter: data_utils.DataLoader, valid_data_iter: data_utils.DataLoader,
                    best_valid_loss: float, saving_path: str):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0

        for i, batch in enumerate(data_iter):
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            model_to_call = self.model.module if hasattr(self.model, "module") else self.model
            mask, target, texts = model_to_call.mask_text(self.mask_prob, batch["pad_mask"], batch["texts"])
            predictions = self.model(device=self.device, mask=mask, texts=texts, pads=batch["pad_mask"])
            ntokens = target.size(0)

            if ntokens == 0:  # Nothing to predict!
                continue

            loss = self.criterion(predictions, target).mean()
            loss.backward()
            if self.optimizer is not None:
                self.optimizer.step()
                self.scheduler.step()

            loss = float(loss.data) * ntokens
            total_loss += loss
            cur_loss += loss
            total_tokens += ntokens
            tokens += ntokens

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print(datetime.datetime.now(),
                      "Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i + 1, cur_loss / tokens, tokens / elapsed))

                if (i + 1) % 5000 == 0:
                    best_valid_loss = self.validate_and_save(best_valid_loss, saving_path, valid_data_iter)

                start, tokens, cur_loss = time.time(), 0, 0

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        best_valid_loss = self.validate_and_save(best_valid_loss, saving_path, valid_data_iter)
        return total_loss / total_tokens, best_valid_loss

    def validate_and_save(self, best_valid_loss, saving_path, valid_data_iter):
        with torch.no_grad():
            total_valid_loss, total_valid_tokens = 0, 0
            for batch in valid_data_iter:
                model_to_call = self.model.module if hasattr(self.model, "module") else self.model
                mask, target, texts = model_to_call.mask_text(self.mask_prob, batch["pad_mask"], batch["texts"].clone())
                predictions = self.model(device=self.device, mask=mask, texts=texts, pads=batch["pad_mask"])
                ntokens = target.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue

                loss = self.criterion(predictions, target).mean().data * ntokens
                total_valid_loss += float(loss)
                total_valid_tokens += ntokens

            valid_loss = total_valid_loss / total_valid_tokens
            print("Current valid loss", valid_loss)
            if best_valid_loss > float(valid_loss):
                best_valid_loss = float(valid_loss)
                print("saving best valid loss", best_valid_loss)
                model_to_save = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )
                model_to_save.save(saving_path)
        return best_valid_loss

    @staticmethod
    def train(options):
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)

        if options.pretrained_path is None:
            lm = LM(text_processor=text_processor, size=options.model_size)
        else:
            lm = LM.load(options.pretrained_path)

        train_data = dataset.TextDataset(save_cache_dir=options.train_cache_path, max_cache_size=options.cache_size)
        valid_data = dataset.TextDataset(save_cache_dir=options.valid_cache_path, max_cache_size=options.cache_size)
        collator = dataset.TextCollator(pad_idx=text_processor.pad_token_id())

        pin_meory = torch.cuda.is_available()
        loader = data_utils.DataLoader(train_data, batch_size=options.batch, shuffle=False, pin_memory=pin_meory,
                                       collate_fn=collator)
        valid_loader = data_utils.DataLoader(valid_data, batch_size=options.batch, shuffle=False, pin_memory=pin_meory,
                                             collate_fn=collator)

        trainer = Trainer(model=lm, mask_prob=options.mask_prob,
                          optimizer=Trainer.build_optimizer(lm.encoder, options.learning_rate, options.weight_decay),
                          clip=options.clip, warmup=options.warmup, warmup_steps=options.warmup_steps)

        best_valid_loss = float("inf")
        for i in range(options.num_epochs):
            print("train epoch", i)
            train_data.current_cache = {} # make sure that we don't use previously masked data!
            _, best_valid_loss = trainer.train_epoch(data_iter=loader, valid_data_iter=valid_loader,
                                                     best_valid_loss=best_valid_loss, saving_path=options.model_path)


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--train_cache", dest="train_cache_path",
                      help="Path to the train data pickle files for large data", metavar="FILE", default=None)
    parser.add_option("--valid_cache", dest="valid_cache_path",
                      help="Path to the train data pickle files for large data", metavar="FILE", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--block", dest="sentence_block", help="Sentence block size", type="int", default=10000)
    parser.add_option("--cache_size", dest="cache_size", help="Number of blocks in cache", type="int", default=100)
    parser.add_option("--vocab_size", dest="vocab_size", help="Vocabulary size", type="int", default=30000)
    parser.add_option("--model", dest="model_path", help="Directory path to save the best model", metavar="FILE",
                      default=None)
    parser.add_option("--pretrained", dest="pretrained_path", help="Directory of pretrained model", metavar="FILE",
                      default=None)
    parser.add_option("--epoch", dest="num_epochs", help="Number of training epochs", type="int", default=25)
    parser.add_option("--clip", dest="clip", help="For gradient clipping", type="int", default=1)
    parser.add_option("--batch", dest="batch", help="Batch size", type="int", default=512)
    parser.add_option("--mask", dest="mask_prob", help="Random masking probability", type="float", default=0.15)
    parser.add_option("--embed", dest="d_model", help="Embedding of contextual word vectors", type="int", default=768)
    parser.add_option("--lr", dest="learning_rate", help="Learning rate", type="float", default=0.0025)
    parser.add_option("--warmup", dest="warmup", help="Warm up rate", type="float", default=0.1)
    parser.add_option("--steps", dest="warmup_steps", help="Number of warmup steps", type="int", default=125000)
    parser.add_option("--decay", dest="weight_decay", help="Weight decay", type="float", default=0.01)
    parser.add_option("--dropout", dest="dropout", help="Dropout probability", type="float", default=0.1)
    parser.add_option("--dff", dest="d_ff", help="Position-wise feed-forward dimensions", type="int", default=2048)
    parser.add_option("--layer", dest="num_layers", help="Number of Layers in cross-attention", type="int", default=2)
    parser.add_option("--heads", dest="num_heads", help="Number of attention heads", type="int", default=8)
    parser.add_option("--freeze", action="store_true", dest="freeze_image", default=False)
    parser.add_option("--size", dest="model_size", help="Model size: 1 (base), 2 (medium), 3 (small)", type="int",
                      default=3)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    print(options)
    Trainer.train(options=options)
