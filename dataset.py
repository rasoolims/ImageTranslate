import logging
import math
import os
import pickle
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, save_cache_dir: str, max_cache_size: int = 100):
        """
        :param save_cache_dir: directory that has saved pickle files for the data.
        :param max_cache_size: Max number of items in cache
        """

        self.current_cache: Dict[Dict[int, torch.LongTensor]] = {}
        self.max_cache_size = max_cache_size
        self.save_cache_dir = save_cache_dir

        with open(os.path.join(save_cache_dir, "info.txt"), "r") as fr:
            spl = fr.read().strip().split("\t")
            self.sentence_block_size = int(spl[0])
            self.line_num = int(spl[1])
            self.file_count = int(spl[2])

    def __len__(self):
        return self.line_num

    def rebuild_cache(self, start_file_num):
        self.current_cache = {}
        for file_num in range(start_file_num, min(self.file_count, start_file_num + self.max_cache_size)):
            with open(os.path.join(self.save_cache_dir, str(file_num)) + ".pkl", "rb") as fp:
                examples = pickle.load(fp)
                self.current_cache[file_num] = examples

    def __getitem__(self, item):
        file_num = math.floor(item / self.sentence_block_size)

        if file_num not in self.current_cache:
            print("Loading data into cache...")
            self.rebuild_cache(file_num)
            print("Loading data into cache done!")
        examples = self.current_cache[file_num]
        return examples[item]


class MTDataset(Dataset):
    def __init__(self, batch_pickle_dir: str, max_batch_capcity: int, max_batch: int, pad_idx: int,
                 max_seq_len: int = 512):
        self.current_cache: Dict[Dict[int, torch.LongTensor]] = {}

        self.batches = []
        with open(batch_pickle_dir, "rb") as fr:
            examples: List[Tuple[torch.tensor, torch.tensor]] = pickle.load(fr)

            cur_src_batch, cur_dst_batch, cur_max_src_len, cur_max_dst_len = [], [], 0, 0
            for example in examples:
                src = example[0][:max_seq_len]  # trim if longer than expected!
                dst = example[1][:max_seq_len]  # trim if longer than expected!

                cur_max_src_len = max(cur_max_src_len, int(src.size(0)))
                cur_max_dst_len = max(cur_max_dst_len, int(dst.size(0)))

                cur_src_batch.append(src)
                cur_dst_batch.append(dst)

                batch_capacity = max(cur_max_src_len ** 2 * len(cur_src_batch),
                                     cur_max_dst_len ** 2 * len(cur_dst_batch))
                batch_size = (cur_max_src_len + cur_max_dst_len) * len(cur_src_batch)

                if batch_capacity > max_batch_capcity or batch_size > max_batch:
                    src_batch = pad_sequence(cur_src_batch[:-1], batch_first=True, padding_value=pad_idx)
                    dst_batch = pad_sequence(cur_dst_batch[:-1], batch_first=True, padding_value=pad_idx)
                    src_pad_mask = (src_batch == pad_idx)
                    dst_pad_mask = (dst_batch == pad_idx)
                    self.batches.append({"src_texts": src_batch, "src_pad_mask": src_pad_mask, "dst_texts": dst_batch,
                                         "dst_pad_mask": dst_pad_mask})
                    cur_src_batch, cur_dst_batch = cur_src_batch[-1], cur_dst_batch[-1]
                    cur_max_src_len, cur_max_dst_len = 0, 0

        if len(cur_src_batch) > 0:
            src_batch = pad_sequence(cur_src_batch, batch_first=True, padding_value=pad_idx)
            dst_batch = pad_sequence(cur_dst_batch, batch_first=True, padding_value=pad_idx)
            src_pad_mask = (src_batch == pad_idx)
            dst_pad_mask = (dst_batch == pad_idx)
            self.batches.append({"src_texts": src_batch, "src_pad_mask": src_pad_mask, "dst_texts": dst_batch,
                                 "dst_pad_mask": dst_pad_mask})

        print("loaded %d bitext sentences to %d batches!" % (len(examples), len(self.batches)))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        return self.batches[item]


class TextCollator(object):
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        padded_text = pad_sequence(batch, batch_first=True, padding_value=self.pad_idx)
        pad_mask = (padded_text == self.pad_idx)
        return {"texts": padded_text, "pad_mask": pad_mask}
