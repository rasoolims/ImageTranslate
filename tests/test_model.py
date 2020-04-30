import os
import tempfile
import unittest
from pathlib import Path

import torch
from torchvision import transforms

import binarize_image_doc_data
import create_batches
from albert_seq2seq import AlbertSeq2Seq
from dataset import TextDataset, ImageDocDataset
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
            processor.train_tokenizer([data_path], vocab_size=1000, to_save_dir=tmpdirname)
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
            processor.train_tokenizer([data_path], vocab_size=1000, to_save_dir=tmpdirname)
            lm = LM(text_processor=processor)
            assert lm.encoder.base_model.embeddings.word_embeddings.num_embeddings == 1000

            lm.save(tmpdirname)

            new_lm = LM.load(tmpdirname)

            assert new_lm.config == lm.config

    def test_albert_seq2seq_init(self):
        path_dir_name = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(path_dir_name, "sample_txt/")
        paths = [str(x) for x in Path(data_path).glob("*.txt")]

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor = TextProcessor()
            processor.train_tokenizer(paths, vocab_size=1000, to_save_dir=tmpdirname)
            lm = LM(text_processor=processor, size=2)

            seq2seq = AlbertSeq2Seq(lm)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            src_inputs = torch.tensor([[1, 2, 3, 4, 5, processor.pad_token_id(), processor.pad_token_id()],
                                       [1, 2, 3, 4, 5, 6, processor.pad_token_id()]])
            tgt_inputs = torch.tensor(
                [[6, 8, 7, processor.pad_token_id(), processor.pad_token_id()], [6, 8, 7, 8, processor.pad_token_id()]])
            src_mask = (src_inputs == processor.pad_token_id())
            tgt_mask = (tgt_inputs == processor.pad_token_id())

            seq_output = seq2seq(device, src_inputs, tgt_inputs, src_mask, tgt_mask, log_softmax=True, flatten=True)
            assert list(seq_output.size()) == [tgt_inputs.size(0) * (tgt_inputs.size(1) - 1), processor.vocab_size()]

            seq_output = seq2seq(device, src_inputs, tgt_inputs, src_mask, tgt_mask)
            assert list(seq_output.size()) == [tgt_inputs.size(0), tgt_inputs.size(1) - 1, processor.vocab_size()]

    def test_data(self):
        path_dir_name = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(path_dir_name, "sample.txt")

        with tempfile.TemporaryDirectory() as tmpdirname, tempfile.TemporaryDirectory() as tmpdirname2:
            processor = TextProcessor()
            processor.train_tokenizer([data_path], vocab_size=1000, to_save_dir=tmpdirname)
            create_batches.write(text_processor=processor, small_cache_dir=tmpdirname, max_small_seq_len=512,
                                 big_cache_dir=tmpdirname2, max_big_seq_len=512,
                                 txt_file=data_path, sentence_small_block_size=10)
            dataset = TextDataset(save_cache_dir=tmpdirname, max_cache_size=3)
            assert dataset.line_num == 92

            dataset.__getitem__(3)
            assert len(dataset.current_cache) == 3

            dataset.__getitem__(9)
            assert len(dataset.current_cache) == 3

            dataset.__getitem__(90)
            assert len(dataset.current_cache) == 1

            dataset.__getitem__(70)
            assert len(dataset.current_cache) == 3

            dataset.__getitem__(80)
            assert len(dataset.current_cache) == 3

            dataset.__getitem__(50)
            assert len(dataset.current_cache) == 3

    def test_image_data(self):
        transform = transforms.Compose([  # [1]
            transforms.Resize(256),  # [2]
            transforms.CenterCrop(224),  # [3]
            transforms.ToTensor(),  # [4]
            transforms.Normalize(  # [5]
                mean=[0.485, 0.456, 0.406],  # [6]
                std=[0.229, 0.224, 0.225]  # [7]
            )])

        path_dir_name = os.path.dirname(os.path.realpath(__file__))
        text_data_path = os.path.join(path_dir_name, "sample.txt")
        data_path = os.path.join(path_dir_name, "image_jsons")

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor = TextProcessor()
            processor.train_tokenizer([text_data_path], vocab_size=1000, to_save_dir=tmpdirname)
            binarize_image_doc_data.write(text_processor=processor, output_file=os.path.join(tmpdirname, "image.bin"),
                                          max_seq_len=512, json_dir=data_path, files_to_use="mzn,glk")
            image_data = ImageDocDataset(os.getcwd(), os.path.join(tmpdirname, "image.bin"), transform, max_doc_batch_capacity=10,
                                         pad_index=processor.pad_token_id())
            assert len(image_data[4])==4
            assert len(image_data) == 21



if __name__ == '__main__':
    unittest.main()
