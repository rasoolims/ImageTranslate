from typing import List, Optional

from tokenizers import Encoding
from tokenizers import SentencePieceBPETokenizer
from tokenizers.processors import BertProcessing


class TextProcessor:
    def __init__(self, tok_model_path: Optional[str] = None):
        self.init_properties()

        if tok_model_path is not None:
            self.tokenizer = SentencePieceBPETokenizer(
                tok_model_path + "/vocab.json",
                tok_model_path + "/merges.txt",
            )
            self.set_postprocess()

    def init_properties(self):
        self.max_len = 512
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.unk_token = "<unk>"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"

    def set_postprocess(self):
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            (self.sep_token, self.sep_token_id()),
            (self.cls_token, self.cls_token_id()),
        )
        self.tokenizer.enable_truncation(max_length=self.max_len)

    def train_tokenizer(self, paths: List[str], vocab_size: int, to_save_dir: str):
        self.tokenizer = SentencePieceBPETokenizer()
        self.init_properties()
        self.tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=2, special_tokens=[
            self.pad_token,
            self.unk_token,
            self.mask_token,
            self.cls_token,
            self.sep_token,
        ])
        self.set_postprocess()
        self.tokenizer.save(directory=to_save_dir)

    def _tokenize(self, line) -> Encoding:
        return self.tokenizer.encode(line)

    def tokenize_one_sentence(self, line) -> List[int]:
        return self._tokenize(line).ids

    def tokenize(self, lines) -> List[List[int]]:
        lines = [line.strip() for line in lines.strip().split("\n") if len(line.strip()) > 0]
        tokenized = self.tokenizer.encode_batch(lines)
        return [tok.ids for tok in tokenized]

    def pad_token_id(self):
        return self.tokenizer.token_to_id(self.pad_token)

    def mask_token_id(self):
        return self.tokenizer.token_to_id(self.mask_token)

    def unk_token_id(self):
        return self.tokenizer.token_to_id(self.unk_token)

    def cls_token_id(self):
        return self.tokenizer.token_to_id(self.cls_token)

    def sep_token_id(self):
        return self.tokenizer.token_to_id(self.sep_token)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 0 for a special token, 1 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id(), self.sep_token_id()] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
