import os
import pickle
import random
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel, AlbertConfig
from transformers.modeling_albert import AlbertMLMHead

from textprocessor import TextProcessor


class LM(nn.Module):
    def __init__(self, text_processor: TextProcessor, config: Dict = None, encoder: AlbertModel = None, size: int = 1):
        """
        :param size: config size: 1 big, 2 medium, 3 small.
        """
        super(LM, self).__init__()
        self.text_processor: TextProcessor = text_processor

        if config is not None:
            self.config = config
        else:
            if size == 1:
                self.config = self._base_config(vocab_size=text_processor.tokenizer.get_vocab_size(),
                                                pad_token_id=text_processor.pad_token_id(),
                                                bos_token_id=text_processor.token_id("<en>"),
                                                eos_token_id=text_processor.token_id("</s>"))
            elif size == 2:
                self.config = self._medium_config(vocab_size=text_processor.tokenizer.get_vocab_size(),
                                                  pad_token_id=text_processor.pad_token_id(),
                                                  bos_token_id=text_processor.token_id("<en>"),
                                                  eos_token_id=text_processor.token_id("</s>"))
            else:
                self.config = self._small_config(vocab_size=text_processor.tokenizer.get_vocab_size(),
                                                 pad_token_id=text_processor.pad_token_id(),
                                                 bos_token_id=text_processor.token_id("<en>"),
                                                 eos_token_id=text_processor.token_id("</s>"))

        albert_config = AlbertConfig(**self.config)
        if encoder is None:
            self.encoder: AlbertModel = AlbertModel(albert_config)
        else:
            self.encoder = encoder

        self.masked_lm = AlbertMLMHead(albert_config)

    def _base_config(self, vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int) -> Dict:
        config = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0.1,
            "embedding_size": 128,
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 16384,
            "max_position_embeddings": 512,
            "num_attention_heads": 64,  # smaller than usual
            "num_hidden_layers": 12,  # smaller than usual
            "num_hidden_groups": 1,
            "net_structure_type": 0,
            "gap_size": 0,
            "num_memory_blocks": 0,
            "inner_group_num": 1,
            "down_scale_factor": 1,
            "type_vocab_size": 2,
            "vocab_size": vocab_size,
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
        }

        return config

    def _medium_config(self, vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int) -> Dict:
        config = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0.1,
            "embedding_size": 128,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 4,  # smaller than usual
            "num_hidden_layers": 4,  # smaller than usual
            "num_hidden_groups": 1,
            "net_structure_type": 0,
            "gap_size": 0,
            "num_memory_blocks": 0,
            "inner_group_num": 1,
            "down_scale_factor": 1,
            "type_vocab_size": 2,
            "vocab_size": vocab_size,
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
        }

        return config

    def _small_config(self, vocab_size: int, pad_token_id: int, bos_token_id: int, eos_token_id: int) -> Dict:
        config = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu_new",
            "hidden_dropout_prob": 0.1,
            "embedding_size": 100,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "max_position_embeddings": 512,
            "num_attention_heads": 2,  # smaller than usual
            "num_hidden_layers": 2,  # smaller than usual
            "num_hidden_groups": 1,
            "net_structure_type": 0,
            "gap_size": 0,
            "num_memory_blocks": 0,
            "inner_group_num": 1,
            "down_scale_factor": 1,
            "type_vocab_size": 2,
            "vocab_size": vocab_size,
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
        }

        return config

    def forward(self, device, data: Dict[str, torch.Tensor], mask_prob: float = 0.15):
        """
        :param data: A minibatch as dictionary that has transformed image and tokenized text as long tensors.
        :return:
        """
        texts = data["texts"].clone().to(device)
        pads = data["pad_mask"].to(device)

        mask, masked_ids, texts = self.mask_text(device, mask_prob, pads, texts)

        text_hidden, text_cls_head = self.encoder(texts, attention_mask=pads)
        output_predictions = F.log_softmax(self.masked_lm(text_hidden[mask]), dim=1)
        return output_predictions, masked_ids

    def mask_text(self, device, mask_prob, pads, texts):
        assert 0 < mask_prob < 1
        mask = torch.empty(texts.size()).uniform_(0, 1) < mask_prob
        mask = mask.to(device)
        mask[pads] = False  # We should not mask pads.
        masked_ids = texts[mask]
        replacements = masked_ids.clone()
        for i in range(len(replacements)):
            r = random.random()
            if r < 0.8:
                replacements[i] = self.text_processor.mask_token_id()
            elif r < 0.9:
                # Replace with another random word.
                random_index = random.randint(len(self.text_processor.special_tokens),
                                              self.text_processor.vocab_size() - 1)
                replacements[i] = random_index
            else:
                # keep the word
                pass
        texts[mask] = replacements
        return mask, masked_ids, texts

    def save(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "config"), "wb") as fp:
            pickle.dump(self.config, fp)

        torch.save(self.state_dict(), os.path.join(out_dir, "model.state_dict"))
        self.text_processor.tokenizer.save(directory=out_dir)

    @staticmethod
    def load(out_dir: str):
        text_processor = TextProcessor(tok_model_path=out_dir)
        with open(os.path.join(out_dir, "config"), "rb") as fp:
            config = pickle.load(fp)
            lm = LM(text_processor=text_processor, config=config)
            lm.load_state_dict(torch.load(os.path.join(out_dir, "model.state_dict")))
            return lm
