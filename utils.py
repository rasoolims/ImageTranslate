import math
import random

import torch
from torch.nn.utils.rnn import pad_sequence

from pytorch_lamb.pytorch_lamb import Lamb
from textprocessor import TextProcessor


def build_optimizer(model, learning_rate, weight_decay):
    return Lamb(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(.9, .999), adam=True)


def mask_text(mask_prob, pads, texts, text_processor: TextProcessor, mask_eos: bool = True):
    assert 0 < mask_prob < 1
    mask = torch.empty(texts.size()).uniform_(0, 1) < mask_prob
    mask[~pads] = False  # We should not mask pads.
    if not mask_eos:
        eos_idx = texts == text_processor.sep_token_id()
        mask[eos_idx] = False  # We should not mask end-of-sentence (usually in case of BART training).

    masked_ids = texts[mask]
    replacements = masked_ids.clone()
    for i in range(len(replacements)):
        r = random.random()
        if r < 0.8:
            replacements[i] = text_processor.mask_token_id()
        elif r < 0.9:
            # Replace with another random word.
            random_index = random.randint(len(text_processor.special_tokens), text_processor.vocab_size() - 1)
            replacements[i] = random_index
        else:
            # keep the word
            pass
    texts[mask] = replacements
    return mask, masked_ids, texts


def unmask_text(mask, masked_ids, texts):
    # Return back the original masked elements!
    texts[mask] = masked_ids


def mass_mask(mask_prob, pad_indices, src_text, text_processor: TextProcessor):
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
    return src_mask, masked_ids, src_text, to_recover, to_recover_pos, mask_idx


def mass_unmask(src_text, src_mask, masked_ids):
    src_text[src_mask] = masked_ids
