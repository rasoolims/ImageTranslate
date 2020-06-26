import math
import os
import random
from typing import Dict

import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from pytorch_lamb.pytorch_lamb import Lamb
from textprocessor import TextProcessor


def build_optimizer(model, learning_rate, weight_decay, use_adam: bool = False):
    if use_adam:
        return AdamInverseSqrtWithWarmup(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))
    else:
        return Lamb(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(.9, .999), adam=True)


def mask_text(mask_prob, pads, texts, text_processor: TextProcessor, mask_eos: bool = True):
    assert 0 < mask_prob < 1
    mask = torch.empty(texts.size()).uniform_(0, 1) < mask_prob
    mask[~pads] = False  # We should not mask pads.
    if not mask_eos:
        eos_idx = texts == text_processor.sep_token_id()
        mask[eos_idx] = False  # We should not mask end-of-sentence (usually in case of BART training).

    masked_ids = texts[mask]
    random_index = lambda: random.randint(len(text_processor.special_tokens), text_processor.vocab_size() - 1)
    rand_select = lambda r, c: text_processor.mask_token_id() if r < 0.8 else (
        random_index() if r < 0.9 else int(masked_ids[c]))
    replacements = list(map(lambda i: rand_select(random.random(), i), range(masked_ids.size(0))))
    texts[mask] = torch.LongTensor(replacements)
    return mask, masked_ids, texts


def unmask_text(mask, masked_ids, texts):
    # Return back the original masked elements!
    texts[mask] = masked_ids


def mass_mask(mask_prob, pad_indices, src_text, text_processor: TextProcessor) -> Dict:
    """
        20% of times, mask from start to middle
        20% of times, mask from middle to end
        60% of times, mask a random index
    """
    index_range = pad_indices - (1 - mask_prob) * pad_indices
    src_mask = torch.zeros(src_text.size(), dtype=torch.bool)
    to_recover = []
    to_recover_pos = []
    for i, irange in enumerate(index_range):
        range_size = int(pad_indices[i] / 2)
        r = random.random()
        last_idx = int(math.ceil(irange))
        if r > 0.8:
            start = 1
        elif r > 0.6:
            start = last_idx
        else:
            start = random.randint(2, last_idx) if last_idx >= 2 else 2

        end = start + range_size
        src_mask[i, start:end] = True
        to_recover.append(src_text[i, start - 1:end])
        to_recover_pos.append(torch.arange(start - 1, end))
    to_recover = pad_sequence(to_recover, batch_first=True, padding_value=text_processor.pad_token_id())
    to_recover_pos = pad_sequence(to_recover_pos, batch_first=True, padding_value=int(src_text.size(-1)) - 1)

    assert 0 < mask_prob < 1
    masked_ids = src_text[:, 1:][src_mask[:, 1:]]
    mask_idx = src_text[src_mask]
    random_index = lambda: random.randint(len(text_processor.special_tokens), text_processor.vocab_size() - 1)
    rand_select = lambda r, c: text_processor.mask_token_id() if r < 0.8 else (
        random_index() if r < 0.9 else int(mask_idx[c]))
    replacements = list(map(lambda i: rand_select(random.random(), i), range(mask_idx.size(0))))
    src_text[src_mask] = torch.LongTensor(replacements)
    return {"src_mask": src_mask, "targets": masked_ids, "src_text": src_text, "to_recover": to_recover,
            "positions": to_recover_pos, "mask_idx": mask_idx}


def mass_unmask(src_text, src_mask, masked_ids):
    src_text[src_mask] = masked_ids


def init_distributed(options):
    if options.fp16:
        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
        torch.distributed.init_process_group(backend='nccl', world_size=torch.cuda.device_count(),
                                             rank=options.local_rank)


class AdamInverseSqrtWithWarmup(optim.Adam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
        decay_factor = lr * sqrt(warmup_updates)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        # linearly warmup for the first warmup_updates
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_updates ** 0.5
        for param_group in self.param_groups:
            param_group['num_updates'] = 0

    def get_lr_for_step(self, num_updates):
        # update learning rate
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * (num_updates ** -0.5)

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group['num_updates'] += 1
            param_group['lr'] = self.get_lr_for_step(param_group['num_updates'])
