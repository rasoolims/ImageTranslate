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

class TextDataLoader(object):
    def __init__(self, text_dataset:TextDataset, batch_size: int, pad_idx):
        self.text_dataset = text_dataset
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(text_dataset)/batch_size)
        self.pad_idx = pad_idx
        self.cur_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_index >= len(self.text_dataset):
            self.cur_index = 0
            raise StopIteration

        low_index = math.floor(self.cur_index / self.text_dataset.sentence_block_size)
        last_item = self.cur_index + self.batch_size
        high_index = math.floor(last_item / self.text_dataset.sentence_block_size)
        batch = []

        for idx in range(low_index, high_index+1):
            if idx not in self.text_dataset.current_cache:
                print("Loading data into cache...")
                self.text_dataset.rebuild_cache(idx)
                print("Loading data into cache done!")
            d = self.text_dataset.current_cache[idx]
            keys = list(d.keys())
            first_key, last_key = keys[0], keys[-1]
            index_offset = self.cur_index - first_key
            start_key = keys[index_offset]
            end_index = min(last_item - start_key, last_key + 1 - start_key) + index_offset
            values = list(d.values())[index_offset:end_index]
            batch += values

        self.cur_index  = last_item
        batch = torch.stack(batch) # We know that everything is prepadded! So we don't do padding
        pad_mask = (batch == self.pad_idx)
        return {"texts": batch, "pad_mask": pad_mask}

    def __len__(self):
        return self.num_batches
