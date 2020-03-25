from typing import List

from tokenizers import ByteLevelBPETokenizer
from tokenizers import Encoding
from tokenizers.processors import BertProcessing


class TextProcessor:
    def __init__(self, tok_model_path: str = None):
        if tok_model_path is not None:
            self.tokenizer = ByteLevelBPETokenizer(
                tok_model_path + "/vocab.json",
                tok_model_path + "/merges.txt",
            )
            self.tokenizer._tokenizer.post_processor = BertProcessing(
                ("</s>", self.tokenizer.token_to_id("</s>")),
                ("<s>", self.tokenizer.token_to_id("<s>")),
            )
            self.tokenizer.enable_truncation(max_length=512)

    def train_tokenizer(self, paths: List[str], vocab_size: int, to_save_dir: str):
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", self.tokenizer.token_to_id("</s>")),
            ("<s>", self.tokenizer.token_to_id("<s>")),
        )
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.save(directory=to_save_dir)

    def tokenize(self, line) -> Encoding:
        return self.tokenizer.encode(line)
