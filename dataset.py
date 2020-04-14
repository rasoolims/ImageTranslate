import logging
import math
import os
import pickle
from typing import Dict

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


class TextCollator(object):
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        padded_text = pad_sequence(batch, batch_first=True, padding_value=self.pad_idx)
        pad_mask = (padded_text == self.pad_idx)
        return {"texts": padded_text, "pad_mask": pad_mask}
