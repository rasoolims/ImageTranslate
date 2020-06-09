import os
import pickle
import sys
from optparse import OptionParser

import torch
import torch.utils.data as data_utils
from IPython.core import ultratb

import dataset
from reformer_lm import ReformerLM
from textprocessor import TextProcessor
from train_lm import LMTrainer

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class ReformerTrainer(LMTrainer):
    @staticmethod
    def train(options):
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)

        if options.pretrained_path is None:
            lm = ReformerLM(text_processor=text_processor, size=options.model_size)
        else:
            lm = ReformerLM.load(options.pretrained_path)

        train_data = dataset.TextDataset(save_cache_dir=options.train_cache_path, max_cache_size=options.cache_size,
                                         load_all=options.distributed)
        valid_data = dataset.TextDataset(save_cache_dir=options.valid_cache_path, max_cache_size=options.cache_size,
                                         load_all=True)

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer, last_epoch = pickle.load(fp)
        else:
            optimizer, last_epoch = ReformerTrainer.build_optimizer(lm, options.learning_rate, options.weight_decay), 0
        trainer = ReformerTrainer(model=lm, mask_prob=options.mask_prob,
                                  optimizer=optimizer,
                                  clip=options.clip, warmup=options.warmup, step=options.step,
                                  fp16=options.fp16, fp16_opt_level=options.fp16_opt_level,
                                  distributed=options.distributed,
                                  local_rank=options.local_rank, last_epoch=last_epoch)

        collator = dataset.TextCollator(pad_idx=text_processor.pad_token_id())
        train_sampler, valid_sampler = None, None
        if options.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)

        pin_memory = torch.cuda.is_available()
        loader = data_utils.DataLoader(train_data, batch_size=options.batch, shuffle=False, pin_memory=pin_memory,
                                       collate_fn=collator, sampler=train_sampler)
        valid_loader = data_utils.DataLoader(valid_data, batch_size=options.batch, shuffle=False, pin_memory=pin_memory,
                                             collate_fn=collator, sampler=valid_sampler)

        step, train_epoch = 0, 1
        while step <= options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(data_iter=loader, valid_data_iter=valid_loader, saving_path=options.model_path,
                                       step=step)


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
    parser.add_option("--warmup", dest="warmup", help="Number of warmup steps", type="int", default=12500)
    parser.add_option("--step", dest="step", help="Number of training steps", type="int", default=125000)
    parser.add_option("--decay", dest="weight_decay", help="Weight decay", type="float", default=0.01)
    parser.add_option("--max_grad_norm", dest="max_grad_norm", help="Max grad norm", type="float", default=1.0)
    parser.add_option("--dropout", dest="dropout", help="Dropout probability", type="float", default=0.1)
    parser.add_option("--dff", dest="d_ff", help="Position-wise feed-forward dimensions", type="int", default=2048)
    parser.add_option("--cont", action="store_true", dest="continue_train",
                      help="Continue training from pretrained model", default=False)
    parser.add_option("--local_rank", dest="local_rank", help="For distributed training", type="int", default=0)
    parser.add_option("--fp16", action="store_true", dest="fp16", help="use fp16; should be compatible", default=False)
    parser.add_option("--distributed", action="store_true", dest="distributed",
                      help="Use distributed data parallelism using the Apex library.", default=False)
    parser.add_option("--size", dest="model_size", help="1: small, 2: medium, 3: base", type="int", default=1)
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
    ReformerTrainer.train(options=options)
    print("Finished Training!")
