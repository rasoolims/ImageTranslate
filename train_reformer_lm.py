import os
import pickle
import sys

import torch
import torch.utils.data as data_utils
from IPython.core import ultratb

import dataset
import train_lm
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

        train_data = dataset.TextDataset(save_cache_dir=options.train_path, max_cache_size=options.cache_size,
                                         load_all=options.distributed)
        dev_data = dataset.TextDataset(save_cache_dir=options.dev_path, max_cache_size=options.cache_size,
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
        train_sampler, dev_sampler = None, None
        if options.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data)

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


def get_options_parser():
    return train_lm.get_option_parser()


if __name__ == "__main__":
    parser = get_options_parser()
    (options, args) = parser.parse_args()
    print(options)
    ReformerTrainer.train(options=options)
    print("Finished Training!")
