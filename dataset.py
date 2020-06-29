import datetime
import glob
import logging
import marshal
import math
import os
import random
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from textprocessor import TextProcessor

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, save_cache_dir: str, max_cache_size: int = 100, load_all: bool = False):
        """
        :param save_cache_dir: directory that has saved marshal files for the data.
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
                examples = marshal.load(fp)
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
                 pad_idx: int, max_seq_len: int = 512, rank: int = -1):
        self.build_batches(batch_pickle_dir, max_batch_capacity, max_batch, pad_idx, max_seq_len, rank)

    def build_batches(self, batch_pickle_dir: str, max_batch_capacity: int, max_batch: int,
                      pad_idx: int, max_seq_len: int = 512, rank: int = -1):
        """
                Since training is fully-batched and has memory/computational need for cubic power of target length, and quadratic
                power of source length, we need to make sure that each batch has similar length and it does not go over
                max_batch_capacity. We also need to make sure not to include those batches that has less than num_gpu
                sentence pairs (it will crash in multi-gpu).
                """
        ngpu = torch.cuda.device_count()
        self.batches = []
        self.longest_batch = ([], 0)
        self.most_token_batch = ([], 0)
        num_gpu = torch.cuda.device_count() if rank == -1 else 1
        paths = glob.glob(batch_pickle_dir + "*")
        for path in paths:
            part_num = int(path[path.rfind(".") + 1:])
            if rank >= 0 and part_num % ngpu != rank:
                continue

            with open(path, "rb") as fr:
                examples: List[Tuple[torch.tensor, torch.tensor]] = marshal.load(fr)

                cur_src_batch, cur_dst_batch, cur_max_src_len, cur_max_dst_len = [], [], 0, 0
                cur_src_langs, cur_dst_langs = [], []
                for example in examples:
                    src = torch.LongTensor(example[0][:max_seq_len])  # trim if longer than expected!
                    dst = torch.LongTensor(example[1][:max_seq_len])  # trim if longer than expected!
                    cur_src_langs.append(example[2])
                    cur_dst_langs.append(example[3])
                    cur_max_src_len = max(cur_max_src_len, int(src.size(0)))
                    cur_max_dst_len = max(cur_max_dst_len, int(dst.size(0)))

                    cur_src_batch.append(src)
                    cur_dst_batch.append(dst)

                    batch_capacity_size = (cur_max_src_len ** 2 + cur_max_dst_len ** 2) * len(
                        cur_src_batch) * cur_max_dst_len
                    batch_size = (cur_max_src_len + cur_max_dst_len) * len(cur_src_batch)

                    if (batch_size > max_batch or batch_capacity_size > max_batch_capacity * 1000000) and \
                            len(cur_src_batch[:-1]) >= num_gpu and len(cur_src_batch) > 1:
                        src_batch = pad_sequence(cur_src_batch[:-1], batch_first=True, padding_value=pad_idx)
                        dst_batch = pad_sequence(cur_dst_batch[:-1], batch_first=True, padding_value=pad_idx)
                        src_pad_mask = (src_batch != pad_idx)
                        dst_pad_mask = (dst_batch != pad_idx)
                        entry = {"src_texts": src_batch, "src_pad_mask": src_pad_mask, "dst_texts": dst_batch,
                                 "dst_pad_mask": dst_pad_mask, "src_langs": torch.LongTensor(cur_src_langs[:-1]),
                                 "dst_langs": torch.LongTensor(cur_dst_langs[:-1])}
                        b, s, d = int(src_batch.size(0)), int(src_batch.size(1)), int(dst_batch.size(1))
                        this_batch_size = (s ** 2 + d ** 2) * b * d
                        if this_batch_size > self.longest_batch[1]:
                            self.longest_batch = (entry, this_batch_size)
                        if b * (s + d) > self.most_token_batch[1]:
                            self.most_token_batch = (entry, b * (s + d))
                        self.batches.append(entry)
                        cur_src_batch, cur_dst_batch = [cur_src_batch[-1]], [cur_dst_batch[-1]]
                        cur_src_langs, cur_dst_langs = [cur_src_langs[-1]], [cur_dst_langs[-1]]
                        cur_max_src_len, cur_max_dst_len = int(cur_src_batch[0].size(0)), int(cur_dst_batch[0].size(0))

        if len(cur_src_batch) > 0 and len(cur_src_batch) >= num_gpu:
            src_batch = pad_sequence(cur_src_batch, batch_first=True, padding_value=pad_idx)
            dst_batch = pad_sequence(cur_dst_batch, batch_first=True, padding_value=pad_idx)
            src_pad_mask = (src_batch != pad_idx)
            dst_pad_mask = (dst_batch != pad_idx)
            entry = {"src_texts": src_batch, "src_pad_mask": src_pad_mask, "dst_texts": dst_batch,
                     "dst_pad_mask": dst_pad_mask, "src_langs": torch.LongTensor(cur_src_langs),
                     "dst_langs": torch.LongTensor(cur_dst_langs)}
            b, s, d = int(src_batch.size(0)), int(src_batch.size(1)), int(dst_batch.size(1))
            this_batch_size = (s ** 2 + d ** 2) * b * d
            if this_batch_size > self.longest_batch[1]:
                self.longest_batch = (entry, this_batch_size)
            if b * (s + d) > self.most_token_batch[1]:
                self.most_token_batch = (entry, b * (s + d))
            self.batches.append(entry)

        for b in self.batches:
            pads = b["src_pad_mask"]
            pad_indices = [int(pads.size(1)) - 1] * int(pads.size(0))
            pindices = torch.nonzero(~pads)
            for (r, c) in pindices:
                pad_indices[r] = min(pad_indices[r], int(c))
            b["pad_idx"] = torch.LongTensor(pad_indices)

        print("Loaded %d bitext sentences to %d batches!" % (len(examples), len(self.batches)))
        print("Longest batch size", self.longest_batch[0]["src_texts"].size(),
              self.longest_batch[0]["dst_texts"].size())
        print("Most token batch size", self.most_token_batch[0]["src_texts"].size(),
              self.most_token_batch[0]["dst_texts"].size())

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        return self.batches[item]


class MassDataset(Dataset):
    def __init__(self, batch_pickle_dir: str, max_batch_capacity: int, max_batch: int,
                 pad_idx: int, max_seq_len: int = 512, keep_examples: bool = False, example_list: List = None,
                 rank: int = -1):
        if example_list is None:
            self.build_batches(batch_pickle_dir, max_batch_capacity, max_batch, pad_idx, max_seq_len, keep_examples,
                               rank)
        else:
            self.examples_list = example_list
            self.batch_items(max_batch, max_batch_capacity, max_seq_len, pad_idx, rank)

    @staticmethod
    def read_example_file(path):
        print(datetime.datetime.now(), "Loading", path)
        with open(path, "rb") as fr:
            examples: List[Tuple[torch.tensor, torch.tensor]] = marshal.load(fr)
        return examples

    def build_batches(self, batch_pickle_dir: str, max_batch_capacity: int, max_batch: int,
                      pad_idx: int, max_seq_len: int = 175, keep_examples: bool = False, rank: int = -1):
        """
        Since training is fully-batched and has memory/computational need for cubic power of target length, and quadratic
        power of source length, we need to make sure that each batch has similar length and it does not go over
        max_batch_capacity. We also need to make sure not to include those batches that has less than num_gpu
        sentence pairs (it will crash in multi-gpu).
        MASS refers to https://arxiv.org/pdf/1905.02450.pdf
        """
        ngpu = torch.cuda.device_count()
        paths = glob.glob(batch_pickle_dir + "*")
        self.examples_list = []
        for path in paths:
            part_num = int(path[path.rfind(".") + 1:])
            if rank >= 0 and part_num % ngpu != rank:
                continue
            self.examples_list.append(MassDataset.read_example_file(path))
        print(datetime.datetime.now(), "Done!")

        self.batch_items(max_batch, max_batch_capacity, max_seq_len, pad_idx)
        if not keep_examples:
            self.examples_list = []

    def batch_items(self, max_batch, max_batch_capacity, max_seq_len, pad_idx, rank):
        print(datetime.datetime.now(), "Building batches")
        self.batches = []
        batches, langs = [], []
        self.lang_ids = set()
        num_gpu = torch.cuda.device_count() if rank == -1 else 1
        cur_src_batch, cur_langs, cur_max_src_len = [], [], 0
        for examples in self.examples_list:
            for example in examples:
                if len(example[0]) > max_seq_len:
                    continue
                src, lang = example[0], example[1]
                self.lang_ids.add(int(src[0]))
                cur_langs.append(lang)

                cur_max_src_len = max(cur_max_src_len, len(src))

                cur_src_batch.append(src)

                batch_capacity_size = 2 * (cur_max_src_len ** 3) * len(cur_src_batch)
                batch_size = 2 * cur_max_src_len * len(cur_src_batch)

                if (batch_size > max_batch or batch_capacity_size > max_batch_capacity * 1000000) and \
                        len(cur_src_batch[:-1]) >= num_gpu and len(cur_langs) > 1:
                    batches.append(cur_src_batch[:-1])
                    langs.append(cur_langs[:-1])
                    cur_src_batch = [cur_src_batch[-1]]
                    cur_langs = [cur_langs[-1]]
                    cur_max_src_len = len(cur_src_batch[0])

        if len(cur_src_batch) > 0:
            if len(cur_src_batch) < num_gpu:
                print("skipping", len(cur_src_batch))
            else:
                batches.append(cur_src_batch)
                langs.append(cur_langs)

        padder = lambda b: pad_sequence(b, batch_first=True, padding_value=pad_idx)
        tensorfier = lambda b: list(map(torch.LongTensor, b))
        entry = lambda b, l: {"src_texts": padder(tensorfier(b)), "langs": torch.LongTensor(l)}
        pad_entry = lambda e: {"src_pad_mask": e["src_texts"] != pad_idx, "src_texts": e["src_texts"],
                               "langs": e["langs"]}

        self.batches = list(map(lambda b, l: pad_entry(entry(b, l)), batches, langs))

        for b in self.batches:
            pads = b["src_pad_mask"]
            pad_indices = [int(pads.size(1)) - 1] * int(pads.size(0))
            pindices = torch.nonzero(~pads)
            for (r, c) in pindices:
                pad_indices[r] = min(pad_indices[r], int(c))
            b["pad_idx"] = torch.LongTensor(pad_indices)

        print("Loaded %d MASS batches!" % (len(self.batches)))
        print("Number of languages", len(self.lang_ids))
        print(datetime.datetime.now(), "Done!")

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        return self.batches[item]


class ImageDocDataset(Dataset):
    def __init__(self, root_img_dir: str, data_bin_file: str, transform, max_doc_batch_capacity: int,
                 text_processor: TextProcessor, max_img_per_batch: int, rank: int = -1):
        self.transform = transform
        self.pad_idx = text_processor.pad_token_id()
        self.batches = {}
        self.root_img_dir = root_img_dir
        max_doc_batch_capacity *= 1000000
        self.images_paths = {}
        self.image_batches = {}
        self.image_queue = {}  # For making sure that the images don't fill up memory!
        ngpu = torch.cuda.device_count()

        print("Start", datetime.datetime.now())
        paths = glob.glob(data_bin_file + "*")
        self.examples_list = []
        for path in paths:
            part_num = int(path[path.rfind(".") + 1:])
            if rank >= 0 and part_num % ngpu != rank:
                continue
            print(rank, "--> Reading", path)
            with open(path, "rb") as fp:
                image_info_dict, unique_images, unique_docs = marshal.load(fp)
                self.languages = list(image_info_dict.keys())
                for lang in self.languages:
                    b, im = self.build_lang_batch(image_info_dict[lang], max_doc_batch_capacity,
                                                  text_processor, unique_docs, unique_images, max_img_per_batch)
                    if lang not in self.batches:
                        self.batches[lang] = b
                        self.images_paths[lang] = im
                    else:
                        self.batches[lang] += b
                        self.images_paths[lang] += im

                    self.image_batches[lang] = {}
                    self.image_queue[lang] = []
                    del image_info_dict[lang]

                del image_info_dict
                del unique_images
                del unique_docs

        print("Loaded %d image batches!" % (len(self.batches)))
        print("End", datetime.datetime.now())

    def build_lang_batch(self, image_info_dict, max_doc_batch_capacity, text_processor, unique_docs, unique_images,
                         max_img_per_batch):
        final_batches, final_image_paths = [], []
        tensorfier = lambda b: list(map(torch.LongTensor, b))
        cur_image_batch, cur_doc_batch, cur_caption_batch, cur_lang_batch, doc_indices, doc_split_sizes = [], [], [], [], [], []
        cur_max_doc_cap = 0
        for image, caption_infos in image_info_dict.items():
            captions = [c[0] for c in caption_infos]
            langs = [
                text_processor.languages["<" + c[1] + ">"] if "<" + c[1] + ">" in text_processor.languages else 0
                for c in caption_infos]
            documents = [c[2] for c in caption_infos]

            for d_i, doc in enumerate(documents):
                for c_i, caption in enumerate(captions):
                    if c_i != d_i and langs[c_i] == langs[d_i]:
                        # Skip different docs with same language.
                        continue

                    docs = unique_docs[doc]
                    doc_len = len(docs) * (len(docs[0]) ** 2)  # based on transformer's memory consumption!
                    batch_size = (49 ** 3 + max(doc_len, cur_max_doc_cap) ** 3) * (len(cur_image_batch) + 1)

                    if cur_max_doc_cap > 0 and (batch_size > max_doc_batch_capacity
                                                or len(cur_image_batch) >= max_img_per_batch):
                        all_docs = pad_sequence(tensorfier(cur_doc_batch), batch_first=True,
                                                padding_value=self.pad_idx)
                        all_captions = pad_sequence(tensorfier(cur_caption_batch), batch_first=True,
                                                    padding_value=self.pad_idx)
                        assert len(doc_indices) == all_docs.size(0)
                        assert len(cur_image_batch) == all_captions.size(0)

                        final_image_paths.append(cur_image_batch)
                        entry = {"docs": all_docs, "captions": all_captions,
                                 "doc_idx": torch.LongTensor(doc_indices), "doc_split": doc_split_sizes,
                                 "langs": torch.LongTensor(cur_lang_batch)}
                        final_batches.append(entry)
                        cur_image_batch, cur_doc_batch, cur_caption_batch, cur_lang_batch, doc_indices, doc_split_sizes = [], [], [], [], [], []
                        cur_max_doc_cap = 0
                    else:
                        cur_max_doc_cap = max(cur_max_doc_cap, doc_len)
                        cur_image_batch.append(unique_images[image])
                        caption_id = len(cur_caption_batch)
                        doc_indices += [caption_id] * len(docs)
                        doc_split_sizes.append(len(docs))
                        cur_caption_batch.append(torch.LongTensor(caption))
                        cur_doc_batch += docs
                        cur_lang_batch += [langs[d_i]] * len(docs)
        if len(cur_image_batch) > 0:
            all_docs = pad_sequence(tensorfier(cur_doc_batch), batch_first=True, padding_value=self.pad_idx)
            all_captions = pad_sequence(tensorfier(cur_caption_batch), batch_first=True, padding_value=self.pad_idx)
            entry = {"docs": all_docs, "captions": all_captions, "images": cur_image_batch,
                     "doc_idx": torch.LongTensor(doc_indices), "doc_split": doc_split_sizes,
                     "langs": torch.LongTensor(cur_lang_batch)}
            final_batches.append(entry)
            final_image_paths.append(cur_image_batch)
        return final_batches, final_image_paths

    def read_transform_images(self, cur_image_batch):
        images = []
        for image_path in cur_image_batch:
            with Image.open(os.path.join(self.root_img_dir, image_path)) as im:
                # make sure not to deal with rgba or grayscale images.
                image = self.transform(im.convert("RGB"))
                images.append(image)
                im.close()
        images = torch.stack(images)
        return images

    def __len__(self):
        # We downgrade the size to the smallest language-specifc batch. In our case, this is usually the shared language.
        return min([len(b) for _, b in self.batches.items()]) * len(self.batches)

    def __getitem__(self, i):
        # From different languages in our data, we pick a random language.
        r = self.languages[random.randint(0, len(self.batches) - 1)]

        # We ignore the item number and actually generate a random index.
        item = random.randint(0, len(self.batches[r]) - 1)

        batch = self.batches[r][item]
        if item not in self.image_batches[r]:
            if len(self.image_batches[r]) >= 60000:
                k = self.image_queue[r].pop(0)
                del self.image_batches[r][k]

            self.image_batches[r][item] = self.read_transform_images(self.images_paths[r][item])
            self.image_queue[r].append(item)

        doc_mask = (batch["docs"] != self.pad_idx)
        caption_mask = (batch["captions"] != self.pad_idx)

        return {"images": self.image_batches[r][item], "captions": batch["captions"], "docs": batch["docs"],
                "doc_mask": doc_mask, "langs": batch["langs"],
                "caption_mask": caption_mask, "doc_idx": batch["doc_idx"], "doc_split": batch["doc_split"]}


class TextCollator(object):
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        langs, batch_text = [], []
        for b in batch:
            batch_text.append(torch.LongTensor(b[0]))
            langs.append(b[1])
        padded_text = pad_sequence(batch_text, batch_first=True, padding_value=self.pad_idx)
        pad_mask = (padded_text != self.pad_idx)
        return {"texts": padded_text, "pad_mask": pad_mask, "langs": torch.LongTensor(langs)}


class ImageTextCollator(object):
    def __call__(self, batch):
        return batch
