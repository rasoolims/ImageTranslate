import os
import tempfile
import unittest
from pathlib import Path

from textprocessor import TextProcessor


class TestTextProcessor(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)

    def test_train(self):
        path_dir_name = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(path_dir_name, "sample_txt/")
        paths = [str(x) for x in Path(data_path).glob("*.txt")]

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor = TextProcessor()
            processor.train_tokenizer(paths, vocab_size=1000, to_save_dir=tmpdirname)
            assert processor.tokenizer.get_vocab_size() == 1000
            sen1 = "Obama signed many landmark bills into law during his first two years in office."
            assert processor.tokenize(sen1) is not None

            new_prcoessor = TextProcessor(tok_model_path=tmpdirname)
            assert new_prcoessor.tokenizer.get_vocab_size() == 1000
            sen2 = "Obama signed many landmark bills into law during his first two years in office."
            assert processor.tokenize(sen2) is not None


if __name__ == '__main__':
    unittest.main()
