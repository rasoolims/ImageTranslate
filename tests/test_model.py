import os
import tempfile
import unittest

import torch

import create_batches
from albert_seq2seq import AlbertSeq2Seq
from dataset import TextDataset
from lm import LM
from textprocessor import TextProcessor


class TestModel(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)

    def test_train_tokenizer(self):
        path_dir_name = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(path_dir_name, "sample.txt")

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor = TextProcessor()
            processor.train_tokenizer([data_path], vocab_size=1000, to_save_dir=tmpdirname, languages={"<en>": 0})
            assert processor.tokenizer.get_vocab_size() == 1000
            sen1 = "Obama signed many landmark bills into law during his first two years in office."
            assert processor._tokenize(sen1) is not None

            many_sens = "\n".join([sen1] * 10)
            assert len(processor.tokenize(many_sens)) == 10

            new_prcoessor = TextProcessor(tok_model_path=tmpdirname)
            assert new_prcoessor.tokenizer.get_vocab_size() == 1000
            sen2 = "Obama signed many landmark bills into law during his first two years in office."
            assert processor._tokenize(sen2) is not None

    def test_albert_init(self):
        path_dir_name = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(path_dir_name, "sample.txt")

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor = TextProcessor()
            processor.train_tokenizer([data_path], vocab_size=1000, to_save_dir=tmpdirname, languages={"<en>": 0})
            lm = LM(text_processor=processor)
            assert lm.encoder.base_model.embeddings.word_embeddings.num_embeddings == 1000

            lm.save(tmpdirname)

            new_lm = LM.load(tmpdirname)

            assert new_lm.config == lm.config

    def test_albert_seq2seq_init(self):
        path_dir_name = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(path_dir_name, "sample.txt")

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor = TextProcessor()
            processor.train_tokenizer([data_path], vocab_size=1000, to_save_dir=tmpdirname,
                                      languages={"<en>": 0, "<fa>": 1})
            seq2seq = AlbertSeq2Seq(text_processor=processor, size=4)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            src_inputs = torch.tensor([[1, 2, 3, 4, 5, processor.pad_token_id(), processor.pad_token_id()],
                                       [1, 2, 3, 4, 5, 6, processor.pad_token_id()]])
            tgt_inputs = torch.tensor(
                [[6, 8, 7, processor.pad_token_id(), processor.pad_token_id()], [6, 8, 7, 8, processor.pad_token_id()]])
            src_mask = (src_inputs != processor.pad_token_id())
            tgt_mask = (tgt_inputs != processor.pad_token_id())
            src_langs = torch.tensor([[0], [0]]).squeeze()
            tgt_langs = torch.tensor([[1], [1]]).squeeze()
            seq_output = seq2seq(src_inputs, tgt_inputs, src_mask, tgt_mask, src_langs, tgt_langs, log_softmax=True)
            assert list(seq_output.size()) == [5, processor.vocab_size()]

            seq_output = seq2seq(src_inputs, tgt_inputs, src_mask, tgt_mask, src_langs, tgt_langs)
            assert list(seq_output.size()) == [5, processor.vocab_size()]

    def test_data(self):
        path_dir_name = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(path_dir_name, "sample.txt")

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor = TextProcessor()
            processor.train_tokenizer([data_path], vocab_size=1000, to_save_dir=tmpdirname,
                                      languages={"<mzn>": 0, "<glk": 1})
            create_batches.write(text_processor=processor, cache_dir=tmpdirname, seq_len=512, txt_file=data_path,
                                 sen_block_size=10)
            dataset = TextDataset(save_cache_dir=tmpdirname, max_cache_size=3)
            assert dataset.line_num == 70

            dataset.__getitem__(3)
            assert len(dataset.current_cache) == 3

            dataset.__getitem__(9)
            assert len(dataset.current_cache) == 3

            dataset.__getitem__(69)
            assert len(dataset.current_cache) == 2


if __name__ == '__main__':
    unittest.main()
