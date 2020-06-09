import os
import pickle
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ReformerModel, ReformerConfig, ReformerModelWithLMHead
from transformers.modeling_reformer import ReformerOnlyLMHead

from reformer_config import _small_config, _medium_config, _base_config
from textprocessor import TextProcessor


class ReformerLM(nn.Module):
    def __init__(self, text_processor: TextProcessor, config: ReformerConfig = None, size: int = 1):
        """
        :param size: config size: 1 small, 2 medium, 3 base.
        """
        super(ReformerLM, self).__init__()
        self.text_processor: TextProcessor = text_processor

        if config is not None:
            self.config = config
        else:
            config_func = _small_config if size == 1 else (_base_config if size == 3 else _medium_config)
            self.config = config_func(vocab_size=text_processor.tokenizer.get_vocab_size(),
                                      pad_token_id=text_processor.pad_token_id(),
                                      eos_token_id=text_processor.sep_token_id())
            self.config = ReformerConfig(**self.config)

        reformer = ReformerModelWithLMHead(self.config)
        self.lm_head: ReformerOnlyLMHead = reformer.lm_head
        self.encoder: ReformerModel = reformer.reformer

    def forward(self, device, mask: torch.Tensor, texts: torch.Tensor, pads: torch.Tensor, langs: List = None):
        """
        We currently don't use langs.
        :param data: A minibatch as dictionary that has transformed image and tokenized text as long tensors.
        :return:
        """
        texts = texts.to(device)
        pads = pads.to(device)
        text_hidden = self.encoder(texts, attention_mask=pads)[0]
        output_predictions = F.log_softmax(self.lm_head(text_hidden[mask]), dim=1)
        return output_predictions

    def save(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "config"), "wb") as fp:
            pickle.dump(self.config, fp)

        torch.save(self.state_dict(), os.path.join(out_dir, "model.state_dict"))
        self.text_processor.save(directory=out_dir)

    @staticmethod
    def load(out_dir: str):
        text_processor = TextProcessor(tok_model_path=out_dir)
        with open(os.path.join(out_dir, "config"), "rb") as fp:
            config = pickle.load(fp)
            if isinstance(config, dict):
                # For older configs
                config = ReformerConfig(**config)
            lm = ReformerLM(text_processor=text_processor, config=config)
            lm.load_state_dict(torch.load(os.path.join(out_dir, "model.state_dict")))
            return lm
