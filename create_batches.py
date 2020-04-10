import os
import pickle
from optparse import OptionParser
from typing import Optional

import torch

from textprocessor import TextProcessor


def write(text_processor: TextProcessor, save_cache_dir: str, max_seq_len: int, txt_file: str,
          sentence_block_size: int = 10000):
    if not os.path.exists(save_cache_dir):
        os.makedirs(save_cache_dir)
    sentence_block_size = sentence_block_size

    current_cache = []
    examples = {}
    line_num, file_count = 0, 0
    with open(os.path.join(txt_file, txt_file), "r") as fp:
        for ln, line in enumerate(fp):
            if len(line.strip()) == 0: continue
            tok_line = text_processor.tokenize_one_line(line.strip())
            tok_lines = text_processor.split_tokenized(tok_line, max_length=max_seq_len)
            current_cache += tok_lines

            if len(current_cache) >= 100000:
                for tok_line in current_cache:
                    # assuming that every list has same length due to correct padding.
                    examples[line_num] = torch.LongTensor(tok_line)
                    line_num += 1
                    if len(examples) >= sentence_block_size:
                        with open(os.path.join(save_cache_dir, str(file_count) + ".pkl"), "wb") as fw:
                            pickle.dump(examples, fw)
                        examples, file_count = {}, file_count + 1
                current_cache = []
                print(f"from {ln} actual documents, dumped {line_num} vectors into {file_count} files")

    if len(current_cache) > 0:
        for tok_line in current_cache:
            examples[line_num] = torch.LongTensor(tok_line)
            line_num += 1
            if len(examples) >= sentence_block_size:
                with open(os.path.join(save_cache_dir, str(file_count) + ".pkl"), "wb") as fw:
                    pickle.dump(examples, fw)
                examples, file_count = {}, file_count + 1
        if len(examples) >= 0:
            with open(os.path.join(save_cache_dir, str(file_count) + ".pkl"), "wb") as fw:
                pickle.dump(examples, fw)
            examples, file_count = {}, file_count + 1

        print(f"from {ln} actual documents, dumped {line_num} vectors into {file_count} files")

    with open(os.path.join(save_cache_dir, "info.txt"), "w") as fw:
        fw.write(str(sentence_block_size) + "\t" + str(line_num) + "\t" + str(file_count))


def get_tokenizer(tokenizer_path: Optional[str] = None, train_path: Optional[str] = None,
                  model_path: Optional[str] = None, vocab_size: Optional[int] = None) -> TextProcessor:
    if tokenizer_path is None:
        print("Training Tokenizer...")
        text_processor = TextProcessor()
        text_processor.train_tokenizer(paths=[train_path], vocab_size=vocab_size, to_save_dir=model_path)
        print("done!")
    else:
        text_processor = TextProcessor(tokenizer_path)
    return text_processor


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--data", dest="data_path", help="Path to the data folder", metavar="FILE", default=None)
    parser.add_option("--cache_big", dest="large_cache_path",
                      help="Path to the data pickle files for large data with large sequence length", metavar="FILE",
                      default=None)
    parser.add_option("--cache_small", dest="small_cache_path",
                      help="Path to the data pickle files for large data", metavar="FILE", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--block", dest="sentence_block", help="Sentence block size", type="int", default=10000)
    parser.add_option("--vocab_size", dest="vocab_size", help="Vocabulary size", type="int", default=30000)
    parser.add_option("--model", dest="model_path", help="Directory path to save the best model", metavar="FILE",
                      default=None)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    tokenizer = get_tokenizer(tokenizer_path=options.tokenizer_path, train_path=options.data_path,
                              model_path=options.model_path, vocab_size=options.vocab_size)

    print("writing batch of 128")
    write(text_processor=tokenizer, save_cache_dir=options.small_cache_path, max_seq_len=128,
          txt_file=options.data_path, sentence_block_size=4 * options.sentence_block)
    print("writing batch of 512")
    write(text_processor=tokenizer, save_cache_dir=options.large_cache_path, max_seq_len=512,
          txt_file=options.data_path, sentence_block_size=options.sentence_block)
    print("finished")
