import datetime
import glob
import logging
import marshal
import math
import os
import random
from itertools import chain
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

from textprocessor import TextProcessor

logger = logging.getLogger(__name__)


def get_lex_suggestions(lex_dict, input_tensor, pad_idx):
    lst = list(set(chain(*map(lambda w: lex_dict[w], input_tensor))))
    if len(lst) == 0:
        lst = [pad_idx]  # Make sure there is at least one item
    return torch.LongTensor(lst)


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
    def __init__(self, max_batch_capacity: int, max_batch: int,
                 pad_idx: int, max_seq_len: int = 175, batch_pickle_dir: str = None,
                 examples: List[Tuple[torch.tensor, torch.tensor, int, int]] = None, lex_dict=None):
        self.lex_dict = lex_dict
        if examples is None:
            self.build_batches(batch_pickle_dir, max_batch_capacity, max_batch, pad_idx, max_seq_len)
        else:
            num_gpu = torch.cuda.device_count()
            self.batch_examples(examples, max_batch, max_batch_capacity, max_seq_len, num_gpu, pad_idx)

    def build_batches(self, batch_pickle_dir: str, max_batch_capacity: int, max_batch: int,
                      pad_idx: int, max_seq_len: int = 175):
        """
        Since training is fully-batched and has memory/computational need for cubic power of target length, and quadratic
        power of source length, we need to make sure that each batch has similar length and it does not go over
        max_batch_capacity. We also need to make sure not to include those batches that has less than num_gpu
        sentence pairs (it will crash in multi-gpu).
        """
        num_gpu = torch.cuda.device_count()
        with open(batch_pickle_dir, "rb") as fr:
            examples: List[Tuple[torch.tensor, torch.tensor, int, int]] = marshal.load(fr)
            self.batch_examples(examples, max_batch, max_batch_capacity, max_seq_len, num_gpu, pad_idx)

    def batch_examples(self, examples, max_batch, max_batch_capacity, max_seq_len, num_gpu, pad_idx):
        self.batches = []
        cur_src_batch, cur_dst_batch, cur_max_src_len, cur_max_dst_len = [], [], 0, 0
        cur_src_langs, cur_dst_langs, cur_lex_cand_batch = [], [], []
        for ei, example in enumerate(examples):
            src = torch.LongTensor(example[0][:max_seq_len])  # trim if longer than expected!
            dst = torch.LongTensor(example[1][:max_seq_len])  # trim if longer than expected!
            cur_src_langs.append(example[2])
            cur_dst_langs.append(example[3])
            cur_max_src_len = max(cur_max_src_len, int(src.size(0)))
            cur_max_dst_len = max(cur_max_dst_len, int(dst.size(0)))

            cur_src_batch.append(src)
            if self.lex_dict is not None:
                lex_cands = get_lex_suggestions(self.lex_dict, src, pad_idx)
                cur_lex_cand_batch.append(lex_cands)
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
                lex_cand_batch = torch.LongTensor([pad_idx])
                if self.lex_dict is not None:
                    lex_cand_batch = pad_sequence(cur_lex_cand_batch[:-1], batch_first=True, padding_value=pad_idx)
                    cur_lex_cand_batch = [cur_lex_cand_batch[-1]]

                entry = {"src_texts": src_batch, "src_pad_mask": src_pad_mask, "dst_texts": dst_batch,
                         "dst_pad_mask": dst_pad_mask, "src_langs": torch.LongTensor(cur_src_langs[:-1]),
                         "dst_langs": torch.LongTensor(cur_dst_langs[:-1]), "proposal": lex_cand_batch}
                self.batches.append(entry)
                cur_src_batch, cur_dst_batch = [cur_src_batch[-1]], [cur_dst_batch[-1]]
                cur_src_langs, cur_dst_langs = [cur_src_langs[-1]], [cur_dst_langs[-1]]
                cur_max_src_len, cur_max_dst_len = int(cur_src_batch[0].size(0)), int(cur_dst_batch[0].size(0))

            if (ei + 1) % 10000 == 0:
                print(ei, "/", len(examples), end="\r")

        if len(cur_src_batch) > 0 and len(cur_src_batch) >= num_gpu:
            src_batch = pad_sequence(cur_src_batch, batch_first=True, padding_value=pad_idx)
            dst_batch = pad_sequence(cur_dst_batch, batch_first=True, padding_value=pad_idx)
            lex_cand_batch = torch.LongTensor([pad_idx])
            if self.lex_dict is not None:
                lex_cand_batch = pad_sequence(cur_lex_cand_batch, batch_first=True, padding_value=pad_idx)
            src_pad_mask = (src_batch != pad_idx)
            dst_pad_mask = (dst_batch != pad_idx)
            entry = {"src_texts": src_batch, "src_pad_mask": src_pad_mask, "dst_texts": dst_batch,
                     "dst_pad_mask": dst_pad_mask, "src_langs": torch.LongTensor(cur_src_langs),
                     "dst_langs": torch.LongTensor(cur_dst_langs), "proposal": lex_cand_batch}
            self.batches.append(entry)
        for b in self.batches:
            pads = b["src_pad_mask"]
            pad_indices = [int(pads.size(1)) - 1] * int(pads.size(0))
            pindices = torch.nonzero(~pads)
            for (r, c) in pindices:
                pad_indices[r] = min(pad_indices[r], int(c))
            b["pad_idx"] = torch.LongTensor(pad_indices)
        print("\nLoaded %d bitext sentences to %d batches!" % (len(examples), len(self.batches)))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        return self.batches[item]


class MassDataset(Dataset):
    def __init__(self, batch_pickle_dir: str, max_batch_capacity: int, max_batch: int,
                 pad_idx: int, max_seq_len: int = 512, keep_examples: bool = False, example_list: List = None,
                 lex_dict=None):
        self.lex_dict = lex_dict
        if example_list is None:
            self.build_batches(batch_pickle_dir, max_batch_capacity, max_batch, pad_idx, max_seq_len, keep_examples)
        else:
            self.examples_list = example_list
            self.batch_items(max_batch, max_batch_capacity, max_seq_len, pad_idx)

    @staticmethod
    def read_example_file(path):
        print(datetime.datetime.now(), "Loading", path)
        with open(path, "rb") as fr:
            examples: List[Tuple[torch.tensor, torch.tensor]] = marshal.load(fr)
        return examples

    def build_batches(self, batch_pickle_dir: str, max_batch_capacity: int, max_batch: int,
                      pad_idx: int, max_seq_len: int = 175, keep_examples: bool = False):
        """
        Since training is fully-batched and has memory/computational need for cubic power of target length, and quadratic
        power of source length, we need to make sure that each batch has similar length and it does not go over
        max_batch_capacity. We also need to make sure not to include those batches that has less than num_gpu
        sentence pairs (it will crash in multi-gpu).
        MASS refers to https://arxiv.org/pdf/1905.02450.pdf
        """

        paths = glob.glob(batch_pickle_dir + "*")
        self.examples_list = [MassDataset.read_example_file(path) for path in paths]
        print(datetime.datetime.now(), "Done!")

        self.batch_items(max_batch, max_batch_capacity, max_seq_len, pad_idx)
        if not keep_examples:
            self.examples_list = []

    def batch_items(self, max_batch, max_batch_capacity, max_seq_len, pad_idx):
        print(datetime.datetime.now(), "Building batches")
        self.batches = []
        batches, langs = [], []
        self.lang_ids = set()
        num_gpu = torch.cuda.device_count()
        cur_src_batch, cur_langs, cur_max_src_len = [], [], 0
        cur_lex_cand_batch = []
        for examples in self.examples_list:
            for example in examples:
                if len(example[0]) > max_seq_len:
                    continue
                src, lang = example[0], example[1]
                self.lang_ids.add(int(src[0]))
                cur_langs.append(lang)

                cur_max_src_len = max(cur_max_src_len, len(src))

                cur_src_batch.append(src)
                if self.lex_dict is not None:
                    lex_cands = get_lex_suggestions(self.lex_dict, src, pad_idx)
                    cur_lex_cand_batch.append(lex_cands)

                batch_capacity_size = 2 * (cur_max_src_len ** 3) * len(cur_src_batch)
                batch_size = 2 * cur_max_src_len * len(cur_src_batch)

                if (batch_size > max_batch or batch_capacity_size > max_batch_capacity * 1000000) and \
                        len(cur_src_batch[:-1]) >= num_gpu and len(cur_langs) > 1:
                    batches.append((cur_src_batch[:-1], cur_lex_cand_batch[:-1] if self.lex_dict is not None else None))
                    langs.append(cur_langs[:-1])
                    cur_src_batch = [cur_src_batch[-1]]
                    cur_langs = [cur_langs[-1]]
                    if self.lex_dict is not None:
                        cur_lex_cand_batch = [cur_lex_cand_batch[-1]]
                    cur_max_src_len = len(cur_src_batch[0])

        if len(cur_src_batch) > 0:
            if len(cur_src_batch) < num_gpu:
                print("skipping", len(cur_src_batch))
            else:
                batches.append((cur_src_batch, cur_lex_cand_batch if self.lex_dict is not None else None))
                langs.append(cur_langs)

        padder = lambda b: pad_sequence(b, batch_first=True, padding_value=pad_idx)
        tensorfier = lambda b: list(map(torch.LongTensor, b))
        entry = lambda b, l: {"src_texts": padder(tensorfier(b[0])),
                              "proposal": padder(tensorfier(b[1])) if b[1] is not None else torch.LongTensor([pad_idx]),
                              "langs": torch.LongTensor(l)}
        pad_entry = lambda e: {"src_pad_mask": e["src_texts"] != pad_idx, "src_texts": e["src_texts"],
                               "langs": e["langs"], "proposal": e["proposal"]}

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


class ImageCaptionDataset(Dataset):
    def __init__(self, root_img_dir: str, data_bin_file: str, max_capacity: int, text_processor: TextProcessor,
                 max_img_per_batch: int, lex_dict=None):
        self.lex_dict = lex_dict
        self.size_transform = transforms.Resize(256)
        self.crop = transforms.CenterCrop(224)
        self.to_tensor = transforms.ToTensor()
        self.img_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.pad_idx = text_processor.pad_token_id()
        self.batches = []
        self.root_img_dir = root_img_dir
        max_capacity *= 1000000
        self.image_batches = []
        self.lang_ids = set()
        num_gpu = torch.cuda.device_count()
        self.all_captions = []

        print("Start", datetime.datetime.now())
        cur_batch, cur_imgs, cur_lex_cand_batch = [], [], []
        cur_max_len = 0
        with open(data_bin_file, "rb") as fp:
            self.unique_images, captions = marshal.load(fp)
            lang_id = text_processor.id2token(captions[0][1][0])
            self.lang_ids.add(int(captions[0][1][0]))
            self.lang = text_processor.languages[lang_id] if lang_id in text_processor.languages else 0
            for caption_info in captions:
                image_id, caption = caption_info
                if self.unique_images[image_id].lower().endswith(".png"):
                    continue
                caption = torch.LongTensor(caption)
                cur_batch.append(caption)
                self.all_captions.append(caption)
                if self.lex_dict is not None:
                    lex_cands = get_lex_suggestions(self.lex_dict, caption, text_processor.pad_token_id())
                    cur_lex_cand_batch.append(lex_cands)

                cur_imgs.append(image_id)
                cur_max_len = max(cur_max_len, len(caption))
                batch_capacity_size = 2 * (cur_max_len ** 3) * len(cur_batch)
                if (len(cur_imgs) > max_img_per_batch or batch_capacity_size > max_capacity) and len(
                        cur_batch[:-1]) >= num_gpu and len(cur_batch) > 1:
                    batch_tensor = pad_sequence(cur_batch[:-1], batch_first=True, padding_value=self.pad_idx)
                    lex_cand_batch = None
                    if self.lex_dict is not None:
                        lex_cand_batch = pad_sequence(cur_lex_cand_batch[:-1], batch_first=True,
                                                      padding_value=self.pad_idx)
                        cur_lex_cand_batch = [cur_lex_cand_batch[-1]]
                    pads = batch_tensor != self.pad_idx
                    pad_indices = [int(pads.size(1)) - 1] * int(pads.size(0))
                    pindices = torch.nonzero(~pads)
                    for (r, c) in pindices:
                        pad_indices[r] = min(pad_indices[r], int(c))

                    self.batches.append((batch_tensor, pads, torch.LongTensor(pad_indices), lex_cand_batch))
                    self.image_batches.append(cur_imgs[:-1])

                    cur_batch = [cur_batch[-1]]
                    cur_imgs = [cur_imgs[-1]]
                    cur_max_len = len(cur_batch[0])

            if len(cur_batch) > 0:
                batch_tensor = pad_sequence(cur_batch, batch_first=True, padding_value=self.pad_idx)
                pads = batch_tensor != self.pad_idx
                pad_indices = [int(pads.size(1)) - 1] * int(pads.size(0))
                lex_cand_batch = None
                if self.lex_dict is not None:
                    lex_cand_batch = pad_sequence(cur_lex_cand_batch, batch_first=True, padding_value=self.pad_idx)

                pindices = torch.nonzero(~pads)
                for (r, c) in pindices:
                    pad_indices[r] = min(pad_indices[r], int(c))

                self.batches.append((batch_tensor, pads, torch.LongTensor(pad_indices), lex_cand_batch))
                self.image_batches.append(cur_imgs)

        print("Loaded %d image batches of %d unique images and %d all captions!" % (
            len(self.batches), len(self.unique_images), len(self.all_captions)))
        print("End", datetime.datetime.now())

    def __len__(self):
        return len(self.batches)

    def get_img(self, path):
        try:
            with Image.open(os.path.join(self.root_img_dir, path)) as im:
                # make sure not to deal with rgba or grayscale images.
                img = im.convert("RGB")
                img = self.crop(self.size_transform(img))
                im.close()
        except:
            print("Corrupted image", path)
            img = Image.new('RGB', (224, 224))
        return img

    def __getitem__(self, item):
        batch, caption_mask, pad_indices, lex_cand_batch = self.batches[item]
        image_batch = list(map(lambda image_id: self.get_img(self.unique_images[image_id]), self.image_batches[item]))

        # We choose fixed negative samples for all batch items.
        img_tensors = torch.stack(list(map(lambda im: self.img_normalize(self.to_tensor(im)), image_batch)))
        num_neg_samples = min(len(self.all_captions), max(30, len(batch)))
        neg_samples = pad_sequence(random.sample(self.all_captions, num_neg_samples), batch_first=True,
                                   padding_value=self.pad_idx)
        neg_mask = (neg_samples != self.pad_idx)
        return {"images": img_tensors, "captions": batch, "pad_idx": pad_indices, "neg": neg_samples,
                "langs": torch.LongTensor([self.lang] * len(batch)), "caption_mask": caption_mask, "neg_mask": neg_mask,
                "proposal": lex_cand_batch}


class ImageDataset(Dataset):
    def __init__(self, root_img_dir: str, max_img_per_batch: int, target_lang: int, first_token: int):
        self.target_lang = target_lang
        self.first_token = first_token
        self.size_transform = transforms.Resize(256)
        self.crop = transforms.CenterCrop(224)
        self.to_tensor = transforms.ToTensor()
        self.img_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.image_batches = []
        print("Start", datetime.datetime.now())
        cur_imgs = []

        image_dir = os.listdir(root_img_dir)
        for img_path in image_dir:
            if img_path.lower().endswith(".png"):
                continue
            cur_imgs.append(os.path.join(root_img_dir, img_path))
            if len(cur_imgs) >= max_img_per_batch:
                self.image_batches.append(cur_imgs)
                cur_imgs = []
        if len(cur_imgs) > 0:
            self.image_batches.append(cur_imgs)

        print("Loaded %d image batches of %d unique images!" % (len(self.image_batches), len(image_dir)))
        print("End", datetime.datetime.now())

    def __len__(self):
        return len(self.image_batches)

    def get_img(self, path):
        try:
            with Image.open(path) as im:
                # make sure not to deal with rgba or grayscale images.
                img = im.convert("RGB")
                img = self.crop(self.size_transform(img))
                im.close()
        except:
            print("Corrupted image", path)
            img = Image.new('RGB', (224, 224))
        return img

    def __getitem__(self, item):
        image_batch = list(map(lambda path: self.get_img(path), self.image_batches[item]))
        first_tokens = torch.LongTensor([self.first_token] * len(image_batch))
        target_lang = torch.LongTensor([self.target_lang] * len(image_batch))
        img_tensors = torch.stack(list(map(lambda im: self.img_normalize(self.to_tensor(im)), image_batch)))
        return {"images": img_tensors, "tgt_langs": target_lang, "first_tokens": first_tokens,
                "paths": self.image_batches[item]}


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
