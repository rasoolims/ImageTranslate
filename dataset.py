import logging
import math
import os
import pickle
from typing import Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from textprocessor import TextProcessor

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, text_processor: TextProcessor, save_cache_dir: str, input_data_dir: str = None,
                 sentence_block_size: int = 10000, max_cache_size: int = 100):
        """
        :param text_processor:
        :param save_cache_dir: directory that has saved pickle files for the data.
        :param input_data_dir:
        :param sentence_block_size: Size of each block of text in RAM
        :param max_cache_size: Max number of items in cache
        """

        self.current_cache: Dict[Dict[int, torch.LongTensor]] = {}
        self.max_cache_size = max_cache_size
        self.save_cache_dir = save_cache_dir

        if input_data_dir is not None:
            assert os.path.isdir(input_data_dir)
            if not os.path.exists(save_cache_dir):
                os.makedirs(save_cache_dir)
            self.sentence_block_size = sentence_block_size

            current_cache = []
            examples = {}
            self.line_num, self.file_count = 0, 0
            for txt_file in os.listdir(input_data_dir):
                # assuming that all files are txt files
                with open(os.path.join(input_data_dir, txt_file), "r") as fp:
                    for line in fp:
                        if len(line.strip()) == 0: continue
                        tok_line = text_processor.tokenize_one_line(line.strip())
                        tok_lines = text_processor.split_tokenized(tok_line)
                        current_cache += tok_lines

                        if len(current_cache) >= 1000000:
                            sorted_list = sorted(current_cache, key=len)
                            for tok_line in sorted_list:
                                examples[self.line_num] = torch.LongTensor(tok_line)
                                self.line_num += 1
                                if len(examples) >= sentence_block_size:
                                    with open(os.path.join(save_cache_dir, str(self.file_count) + ".pkl"), "wb") as fw:
                                        pickle.dump(examples, fw)
                                    examples, self.file_count = {}, self.file_count + 1
                            current_cache = []
                            print("dumped", self.line_num, "lines into", self.file_count, "files")

            if len(current_cache) > 0:
                sorted_list = sorted(current_cache, key=len)
                for tok_line in sorted_list:
                    examples[self.line_num] = torch.LongTensor(tok_line)
                    self.line_num += 1
                    if len(examples) >= sentence_block_size:
                        with open(os.path.join(save_cache_dir, str(self.file_count) + ".pkl"), "wb") as fw:
                            pickle.dump(examples, fw)
                        examples, self.file_count = {}, self.file_count + 1
                if len(examples) >= 0:
                    with open(os.path.join(save_cache_dir, str(self.file_count) + ".pkl"), "wb") as fw:
                        pickle.dump(examples, fw)
                    examples, self.file_count = {}, self.file_count + 1

                print("Finished saving", self.line_num, "lines into", self.file_count, "files")

            with open(os.path.join(save_cache_dir, "info.txt"), "w") as fw:
                fw.write(str(sentence_block_size) + "\t" + str(self.line_num))
        else:
            with open(os.path.join(save_cache_dir, "info.txt"), "r") as fr:
                spl = fr.read().strip().split("\t")
                self.sentence_block_size = int(spl[0])
                self.line_num = int(spl[1])

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
            self.rebuild_cache(file_num)
        examples = self.current_cache[file_num]
        return examples[item]


class TextCollator(object):
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        padded_text = pad_sequence(batch, batch_first=True, padding_value=self.pad_idx)
        pad_mask = (padded_text == self.pad_idx)
        return {"texts": padded_text, "pad_mask": pad_mask}
