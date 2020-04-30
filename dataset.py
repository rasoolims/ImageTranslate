import logging
import math
import os
import pickle
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, save_cache_dir: str, max_cache_size: int = 100, load_all: bool = False):
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

        if load_all:
            print("Loading all data at once...")
            self.rebuild_cache(0, self.file_count)
            print("Done!")

    def __len__(self):
        return self.line_num

    def rebuild_cache(self, start_file_num, end_file_num):
        self.current_cache = {}
        for file_num in range(start_file_num, end_file_num):
            with open(os.path.join(self.save_cache_dir, str(file_num)) + ".pkl", "rb") as fp:
                examples = pickle.load(fp)
                self.current_cache[file_num] = examples

    def __getitem__(self, item):
        file_num = math.floor(item / self.sentence_block_size)

        if file_num not in self.current_cache:
            print("Loading data into cache...")
            self.rebuild_cache(file_num, min(self.file_count, file_num + self.max_cache_size))
            print("Loading data into cache done!")
        examples = self.current_cache[file_num]
        return examples[item]


class MTDataset(Dataset):
    def __init__(self, batch_pickle_dir: str, max_batch_capacity: int, max_batch: int,
                 pad_idx: int, max_seq_len: int = 512):
        """
        Since training is fully-batched and has memory/computational need for cubic power of target length, and quadratic
        power of source length, we need to make sure that each batch has similar length and it does not go over
        max_batch_capacity. We also need to make sure not to include those batches that has less than num_gpu
        sentence pairs (it will crash in multi-gpu).
        """
        self.batches = []
        self.longest_batch = ([], 0)
        self.most_token_batch = ([], 0)
        num_gpu = torch.cuda.device_count()
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

                batch_capacity_size = (cur_max_src_len ** 2 + cur_max_dst_len ** 2) * len(
                    cur_src_batch) * cur_max_dst_len
                batch_size = (cur_max_src_len + cur_max_dst_len) * len(cur_src_batch)

                if batch_size > max_batch or batch_capacity_size > max_batch_capacity * 1000000:
                    src_batch = pad_sequence(cur_src_batch[:-1], batch_first=True, padding_value=pad_idx)
                    dst_batch = pad_sequence(cur_dst_batch[:-1], batch_first=True, padding_value=pad_idx)
                    src_pad_mask = (src_batch != pad_idx)
                    dst_pad_mask = (dst_batch != pad_idx)
                    if src_batch.size(0) < num_gpu:
                        print("skipping", src_batch.size(), dst_batch.size())
                    else:
                        entry = {"src_texts": src_batch, "src_pad_mask": src_pad_mask, "dst_texts": dst_batch,
                                 "dst_pad_mask": dst_pad_mask}
                        b, s, d = int(src_batch.size(0)), int(src_batch.size(1)), int(dst_batch.size(1))
                        this_batch_size = (s ** 2 + d ** 2) * b * d
                        if this_batch_size > self.longest_batch[1]:
                            self.longest_batch = (entry, this_batch_size)
                        if b * (s + d) > self.most_token_batch[1]:
                            self.most_token_batch = (entry, b * (s + d))
                        self.batches.append(entry)
                    cur_src_batch, cur_dst_batch = [cur_src_batch[-1]], [cur_dst_batch[-1]]
                    cur_max_src_len, cur_max_dst_len = int(cur_src_batch[0].size(0)), int(cur_dst_batch[0].size(0))

        if len(cur_src_batch) > 0:
            src_batch = pad_sequence(cur_src_batch, batch_first=True, padding_value=pad_idx)
            dst_batch = pad_sequence(cur_dst_batch, batch_first=True, padding_value=pad_idx)
            src_pad_mask = (src_batch != pad_idx)
            dst_pad_mask = (dst_batch != pad_idx)
            if src_batch.size(0) < num_gpu:
                print("skipping", src_batch.size(), dst_batch.size())
            else:
                entry = {"src_texts": src_batch, "src_pad_mask": src_pad_mask, "dst_texts": dst_batch,
                         "dst_pad_mask": dst_pad_mask}
                b, s, d = int(src_batch.size(0)), int(src_batch.size(1)), int(dst_batch.size(1))
                this_batch_size = (s ** 2 + d ** 2) * b * d
                if this_batch_size > self.longest_batch[1]:
                    self.longest_batch = (entry, this_batch_size)
                if b * (s + d) > self.most_token_batch[1]:
                    self.most_token_batch = (entry, b * (s + d))
                self.batches.append(entry)

        print("loaded %d bitext sentences to %d batches!" % (len(examples), len(self.batches)))
        print("Longest batch size", self.longest_batch[0]["src_texts"].size(),
              self.longest_batch[0]["dst_texts"].size())
        print("Most token batch size", self.most_token_batch[0]["src_texts"].size(),
              self.most_token_batch[0]["dst_texts"].size())

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        return self.batches[item]


class ImageDocDataset(Dataset):
    def __init__(self, root_img_dir: str, data_bin_file: str, transform, max_doc_batch_capacity: int, pad_index: int):
        self.transform = transform

        self.batches = []
        self.root_img_dir = root_img_dir
        max_doc_batch_capacity *= 1000000
        with open(data_bin_file, "rb") as fp:
            doc_ids, images, captions = pickle.load(fp)
            cur_image_batch, cur_doc_batch, cur_caption_batch, doc_indices = [], [], [], []
            cur_max_doc_cap = 0

            # We can sort the captions based on their corresponding document sentence length (second dim) to
            # maximize GPU utilization.
            caption_len_dict = {id: doc_ids[captions[id]["doc_id"]][0].size(0) for id in captions.keys()}
            sorted_caption_lens = sorted(caption_len_dict.items(), key=lambda x: x[1])

            for id, _ in sorted_caption_lens:
                docs = doc_ids[captions[id]["doc_id"]]
                caption = captions[id]["caption"]
                doc_len = len(docs) * (docs[0].size(0) ** 2)  # based on transformer's memory consumption!

                if cur_max_doc_cap > 0 and max(cur_max_doc_cap, doc_len) * (
                        len(cur_caption_batch) + 1) > max_doc_batch_capacity:
                    all_docs = pad_sequence(cur_doc_batch, batch_first=True, padding_value=pad_index)
                    all_captions = pad_sequence(cur_caption_batch, batch_first=True, padding_value=pad_index)
                    assert len(doc_indices) == all_docs.size(0)
                    assert len(cur_image_batch) == all_captions.size(0)
                    entry = {"docs": all_docs, "captions": all_captions, "images": cur_image_batch,
                             "doc_idx": doc_indices}
                    self.batches.append(entry)
                    cur_image_batch, cur_doc_batch, cur_caption_batch, doc_indices = [], [], [], []
                    cur_max_doc_cap = 0
                else:
                    cur_max_doc_cap = max(cur_max_doc_cap, doc_len)
                    cur_image_batch.append(images[captions[id]["image_id"]])
                    caption_id = len(cur_caption_batch)
                    doc_indices += [caption_id] * len(docs)
                    cur_caption_batch.append(torch.LongTensor(caption))
                    cur_doc_batch += docs

            if len(cur_image_batch) > 0:
                all_docs = pad_sequence(cur_doc_batch, batch_first=True, padding_value=pad_index)
                all_captions = pad_sequence(cur_caption_batch, batch_first=True, padding_value=pad_index)
                entry = {"docs": all_docs, "captions": all_captions, "images": cur_image_batch, "doc_idx": doc_indices}
                self.batches.append(entry)

            print("loaded %d images, %d captions, %d docs in %d batches!" % (
                len(images), len(captions), len(docs), len(self.batches)))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        batch = self.batches[item]
        images = []
        for image_path in batch["images"]:  # todo do it parallel?
            # make sure not to deal with rgba or grayscale images.
            image = Image.open(os.path.join(self.root_img_dir, image_path)).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        return {"images": images, "caption": batch["caption"], "doc": batch["doc"], "doc_idx": batch["doc_idx"]}


class TextCollator(object):
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        padded_text = pad_sequence(batch, batch_first=True, padding_value=self.pad_idx)
        pad_mask = (padded_text != self.pad_idx)
        return {"texts": padded_text, "pad_mask": pad_mask}
