from typing import List

from tokenizers import ByteLevelBPETokenizer
from tokenizers import Encoding
from tokenizers.processors import BertProcessing


class TextProcessor:
    def __init__(self):
        self.tokenizer = ByteLevelBPETokenizer()

    def train_tokenizer(self, paths: List[str], vocab_size: int, to_save_dir: str, model_name: str):
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])
        self.tokenizer.save(directory=to_save_dir, name=model_name)
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", self.tokenizer.token_to_id("</s>")),
            ("<s>", self.tokenizer.token_to_id("<s>")),
        )
        self.tokenizer.enable_truncation(max_length=512)

    def tokenize(self, line) -> Encoding:
        return self.tokenizer.encode(line)
