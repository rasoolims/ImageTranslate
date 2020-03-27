import os
import pickle
from typing import Optional

import torch
from transformers import AlbertModel, AlbertConfig

from textprocessor import TextProcessor


class LM:
    def __init__(self, text_processor: TextProcessor, model: Optional[AlbertModel] = None):
        self.text_processor: TextProcessor = text_processor
        if model is None:
            self.config = self._config(vocab_size=text_processor.tokenizer.get_vocab_size())
            self.model: AlbertModel = AlbertModel(self.config)
        else:
            self.model = model

    def _config(self, vocab_size: int) -> AlbertConfig:
        config = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
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
            "vocab_size": vocab_size
        }

        return AlbertConfig(**config)

    def save(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "config"), "wb") as fp:
            pickle.dump(self.config, fp)

        torch.save(self.model.state_dict(), os.path.join(out_dir, "model.state_dict"))
        self.text_processor.tokenizer.save(directory=out_dir)

    @staticmethod
    def load(out_dir: str):
        text_processor = TextProcessor(tok_model_path=out_dir)
        with open(os.path.join(out_dir, "config"), "rb") as fp:
            config = pickle.load(fp)
            model = AlbertModel(config)
            model.load_state_dict(torch.load(os.path.join(out_dir, "model.state_dict")))
            return LM(text_processor=text_processor, model=model)
