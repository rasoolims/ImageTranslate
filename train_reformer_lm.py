import os
import pickle
import sys

import torch
import torch.utils.data as data_utils
from IPython.core import ultratb

import dataset
from option_parser import get_lm_option_parser
from reformer_lm import ReformerLM
from textprocessor import TextProcessor
from train_lm import LMTrainer
from utils import build_optimizer

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

        train_data = dataset.TextDataset(save_cache_dir=options.train_path, max_cache_size=options.cache_size)
        dev_data = dataset.TextDataset(save_cache_dir=options.dev_path, max_cache_size=options.cache_size,
                                       load_all=True)

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer, last_epoch = pickle.load(fp)
        else:
            optimizer, last_epoch = build_optimizer(lm, options.learning_rate, options.weight_decay,
                                                    use_adam=options.adam), 0

        lm.config.hidden_dropout_prob = options.dropout
        lm.config.local_attention_probs_dropout_prob = options.dropout
        lm.config.lsh_attention_probs_dropout_prob = options.dropout

        trainer = ReformerTrainer(model=lm, mask_prob=options.mask_prob, optimizer=optimizer, clip=options.clip,
                                  warmup=options.warmup, step=options.step, last_epoch=last_epoch)

        collator = dataset.TextCollator(pad_idx=text_processor.pad_token_id())
        train_sampler, dev_sampler = None, None

        pin_memory = torch.cuda.is_available()
        loader = data_utils.DataLoader(train_data, batch_size=options.batch, shuffle=False, pin_memory=pin_memory,
                                       collate_fn=collator, sampler=train_sampler)
        dev_loader = data_utils.DataLoader(dev_data, batch_size=options.batch, shuffle=False, pin_memory=pin_memory,
                                           collate_fn=collator, sampler=dev_sampler)
        step, train_epoch = last_epoch, 1
        while step <= options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(data_iter=loader, dev_data_iter=dev_loader, saving_path=options.model_path,
                                       step=step)


if __name__ == "__main__":
    parser = get_lm_option_parser()
    (options, args) = parser.parse_args()
    print(options)
    ReformerTrainer.train(options=options)
    print("Finished Training!")
