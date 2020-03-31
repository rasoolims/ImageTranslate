import os
import sys
import time
from optparse import OptionParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from IPython.core import ultratb

import dataset
from lm import LM
from textprocessor import TextProcessor

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class NoamOpt:
    "Optim wrapper that implements rate."

    """
    from https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class MaskLoss:
    def __init__(self, model: LM, optimizer=None, clip: int = 1):
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.optimizer = optimizer
        self.clip = clip
        self.model = model

    def __call__(self, prediction, gold_standard, norm):
        loss = self.criterion(prediction.contiguous().view(-1, prediction.size(-1)),
                              gold_standard.contiguous().view(-1))
        loss.backward()
        if self.optimizer is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()
        return loss.data * norm


class Trainer:
    def __init__(self, model: LM, mask_prob: float = 0.15, clip: int = 1, optimizer=None):
        self.model = model
        self.loss_compute = MaskLoss(model, optimizer=optimizer, clip=clip)
        self.mask_prob = mask_prob

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            print("Let's use", num_gpu, "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    @staticmethod
    def get_std_opt(model):
        return NoamOpt(model.config.hidden_size, 2, 4000,
                       torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))

    def train_epoch(self, data_iter: data_utils.DataLoader, valid_data_iter: data_utils.DataLoader,
                    best_valid_loss: float, saving_path: str):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0

        for i, batch in enumerate(data_iter):
            predictions, target = self.model(device=self.device, data=batch, mask_prob=self.mask_prob)
            ntokens = target.size(0)

            if ntokens == 0:  # Nothing to predict!
                continue

            loss = self.loss_compute(predictions, target, ntokens)
            total_loss += loss
            cur_loss += loss
            total_tokens += ntokens
            tokens += ntokens

            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i + 1, cur_loss / tokens, tokens / elapsed))

                if (i + 1) % 5000 == 0:
                    best_valid_loss = self.validate_and_save(best_valid_loss, saving_path, valid_data_iter)

                start, tokens, cur_loss = time.time(), 0, 0

        best_valid_loss = self.validate_and_save(best_valid_loss, saving_path, valid_data_iter)

        return total_loss / total_tokens, best_valid_loss

    def validate_and_save(self, best_valid_loss, saving_path, valid_data_iter):
        with torch.no_grad():
            total_valid_loss, total_valid_tokens = 0, 0
            for batch in valid_data_iter:
                predictions, target = self.model(device=self.device, data=batch, mask_prob=self.mask_prob)
                ntokens = target.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue

                loss = self.loss_compute.criterion(predictions.contiguous().view(-1, predictions.size(-1)),
                                                   target.contiguous().view(-1))
                total_valid_loss += loss
                total_valid_tokens += ntokens

            valid_loss = total_valid_loss / total_valid_tokens
            print("Current valid loss", valid_loss.data)
            if best_valid_loss > float(valid_loss.data):
                best_valid_loss = float(valid_loss.data)
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

        if options.tokenizer_path is None:
            print("Training Tokenizer...")
            text_processor = TextProcessor()
            paths = [str(x) for x in Path(options.train_path).glob("*.txt")]
            text_processor.train_tokenizer(paths=paths, vocab_size=options.vocab_size, to_save_dir=options.model_path)
            print("done!")
        else:
            text_processor = TextProcessor(options.tokenizer_path)
        lm = LM(text_processor=text_processor)

        train_data = dataset.TextDataset(text_processor=text_processor, save_cache_dir=options.train_cache_path,
                                         input_data_dir=options.train_path,
                                         sentence_block_size=options.sentence_block, max_cache_size=options.cache_size)
        valid_data = dataset.TextDataset(text_processor=text_processor, save_cache_dir=options.valid_cache_path,
                                         input_data_dir=options.valid_path,
                                         sentence_block_size=options.sentence_block, max_cache_size=options.cache_size)
        collator = dataset.TextCollator(pad_idx=text_processor.pad_token_id())

        loader = data_utils.DataLoader(train_data, batch_size=options.batch, shuffle=False, collate_fn=collator)
        valid_loader = data_utils.DataLoader(valid_data, batch_size=options.batch, shuffle=False, collate_fn=collator)

        trainer = Trainer(model=lm, mask_prob=options.mask_prob, optimizer=Trainer.get_std_opt(lm.encoder),
                          clip=options.clip)

        best_valid_loss = float("inf")
        for i in range(options.num_epochs):
            with torch.autograd.detect_anomaly():
                _, best_valid_loss = trainer.train_epoch(data_iter=loader, valid_data_iter=valid_loader,
                                                         best_valid_loss=best_valid_loss,
                                                         saving_path=options.model_path)


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--train", dest="train_path", help="Path to the train data folder", metavar="FILE", default=None)
    parser.add_option("--train_cache", dest="train_cache_path",
                      help="Path to the train data pickle files for large data", metavar="FILE", default=None)
    parser.add_option("--valid_cache", dest="valid_cache_path",
                      help="Path to the train data pickle files for large data", metavar="FILE", default=None)
    parser.add_option("--valid", dest="valid_path", help="Path to the dev data folder", metavar="FILE", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--block", dest="sentence_block", help="Sentence block size", type="int", default=10000)
    parser.add_option("--cache_size", dest="cache_size", help="Number of blocks in cache", type="int", default=100)
    parser.add_option("--vocab_size", dest="vocab_size", help="Vocabulary size", type="int", default=30000)
    parser.add_option("--model", dest="model_path", help="Directory path to save the best model", metavar="FILE",
                      default=None)
    parser.add_option("--epoch", dest="num_epochs", help="Number of training epochs", type="int", default=25)
    parser.add_option("--clip", dest="clip", help="For gradient clipping", type="int", default=1)
    parser.add_option("--batch", dest="batch", help="Batch size", type="int", default=512)
    parser.add_option("--mask", dest="mask_prob", help="Random masking probability", type="float", default=0.15)
    parser.add_option("--embed", dest="d_model", help="Embedding of contextual word vectors", type="int", default=768)
    parser.add_option("--dropout", dest="dropout", help="Dropout probability", type="float", default=0.1)
    parser.add_option("--dff", dest="d_ff", help="Position-wise feed-forward dimensions", type="int", default=2048)
    parser.add_option("--layer", dest="num_layers", help="Number of Layers in cross-attention", type="int", default=2)
    parser.add_option("--heads", dest="num_heads", help="Number of attention heads", type="int", default=8)
    parser.add_option("--freeze", action="store_true", dest="freeze_image", default=False)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    Trainer.train(options=options)
