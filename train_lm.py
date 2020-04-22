import datetime
import os
import sys
import time
from collections import defaultdict
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
                 warmup_steps: int = 125000, fp16: bool = False, fp16_opt_level: str = "01"):
        self.model = model

        self.clip = clip
        self.optimizer = optimizer
        self.fp16 = fp16

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if fp16:
            try:
                import apex
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level=fp16_opt_level)

        if self.optimizer is not None:
            self.scheduler = optim.get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=int(warmup * warmup_steps), num_training_steps=warmup_steps
            )
        self.mask_prob = mask_prob
        self.criterion = nn.NLLLoss(ignore_index=model.text_processor.pad_token_id())

        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            print("Let's use", num_gpu, "GPUs!")
            self.model = DataParallelModel(self.model)
            self.criterion = DataParallelCriterion(self.criterion)

        self.best_valid_loss = float("inf")
        self.best_train_loss = float("inf")
        self.last_train_loss = float("inf")

    @staticmethod
    def build_optimizer(model, learning_rate, weight_decay):
        return Lamb(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(.9, .999), adam=True)

    def reset_optimizer(self):
        self.optimizer.state = defaultdict(dict)
        self.scheduler.last_epoch = -1

    def train_epoch(self, data_iter: data_utils.DataLoader, valid_data_iter: data_utils.DataLoader,
                    saving_path: str, max_grad_norm: float = 1.0):
        if self.fp16:
            try:
                import apex
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

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
            if self.fp16:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.optimizer is not None:
                if self.fp16:
                    torch.nn.utils.clip_grad_norm_(apex.amp.master_params(self.optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

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
                    self.validate_and_save(saving_path, valid_data_iter)

                start, tokens, cur_loss = time.time(), 0, 0

        current_loss = total_loss / total_tokens
        print("Total loss in this epoch: %f" % current_loss)
        if current_loss < self.best_train_loss:
            self.best_train_loss = current_loss
            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            model_to_save.save(saving_path + ".latest")
        elif current_loss > self.last_train_loss:
            # Restart optimizer state to see if anything changes
            print("Restarting optimizer!")
            self.reset_optimizer()

        self.last_train_loss = current_loss

        self.validate_and_save(saving_path, valid_data_iter)

    def validate_and_save(self, saving_path, valid_data_iter):
        with torch.no_grad():
            model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            model.eval()
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
            if self.best_valid_loss > float(valid_loss):
                self.best_valid_loss = float(valid_loss)
                print("saving best valid loss", self.best_valid_loss)
                model_to_save = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )
                model_to_save.save(saving_path)
            model.train()

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

        pin_memory = torch.cuda.is_available()
        loader = data_utils.DataLoader(train_data, batch_size=options.batch, shuffle=False, pin_memory=pin_memory,
                                       collate_fn=collator)
        valid_loader = data_utils.DataLoader(valid_data, batch_size=options.batch, shuffle=False, pin_memory=pin_memory,
                                             collate_fn=collator)

        trainer = Trainer(model=lm, mask_prob=options.mask_prob,
                          optimizer=Trainer.build_optimizer(lm.encoder, options.learning_rate, options.weight_decay),
                          clip=options.clip, warmup=options.warmup, warmup_steps=options.warmup_steps,
                          fp16=options.fp16, fp16_opt_level=options.fp16_opt_level)

        if options.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        for i in range(options.num_epochs):
            print("train epoch", i)
            train_data.current_cache = {}  # make sure that we don't use previously masked data!
            trainer.train_epoch(data_iter=loader, valid_data_iter=valid_loader, saving_path=options.model_path)


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--train_cache", dest="train_cache_path",
                      help="Path to the train data pickle files for large data", metavar="FILE", default=None)
    parser.add_option("--valid_cache", dest="valid_cache_path",
                      help="Path to the train data pickle files for large data", metavar="FILE", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--cache_size", dest="cache_size", help="Number of blocks in cache", type="int", default=300)
    parser.add_option("--vocab_size", dest="vocab_size", help="Vocabulary size", type="int", default=30000)
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
    parser.add_option("--warmup", dest="warmup", help="Warm up rate", type="float", default=0.1)
    parser.add_option("--steps", dest="warmup_steps", help="Number of warmup steps", type="int", default=125000)
    parser.add_option("--decay", dest="weight_decay", help="Weight decay", type="float", default=0.01)
    parser.add_option("--max_grad_norm", dest="max_grad_norm", help="Max grad norm", type="float", default=1.0)
    parser.add_option("--dropout", dest="dropout", help="Dropout probability", type="float", default=0.1)
    parser.add_option("--dff", dest="d_ff", help="Position-wise feed-forward dimensions", type="int", default=2048)
    parser.add_option("--layer", dest="num_layers", help="Number of Layers in cross-attention", type="int", default=2)
    parser.add_option("--heads", dest="num_heads", help="Number of attention heads", type="int", default=8)
    parser.add_option("--fp16", action="store_true", dest="fp16", help="use fp16; should be compatible", default=False)
    parser.add_option("--size", dest="model_size", help="Model size: 1 (base), 2 (medium), 3 (small)", type="int",
                      default=3)
    parser.add_option(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    print(options)
    Trainer.train(options=options)
