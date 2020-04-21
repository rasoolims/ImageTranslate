import os
import pickle
from optparse import OptionParser
from typing import Optional

import torch

from textprocessor import TextProcessor


def write(text_processor: TextProcessor, small_cache_dir: str, big_cache_dir: str, max_small_seq_len: int,
          max_big_seq_len: int, txt_file: str,
          sentence_small_block_size: int = 40000, sentence_big_block_size: int = 10000):
    if not os.path.exists(small_cache_dir):
        os.makedirs(small_cache_dir)
    sentence_small_block_size = sentence_small_block_size
    current_small_cache = []
    small_examples = {}
    small_line_num, small_file_count = 0, 0

    sentence_big_block_size = sentence_big_block_size
    current_big_cache = []
    big_examples = {}
    big_line_num, big_file_count = 0, 0

    with open(txt_file, "r") as fp:
        for ln, line in enumerate(fp):
            if len(line.strip()) == 0: continue
            tok_line = text_processor.tokenize_one_line(line.strip())
            small_tok_lines = text_processor.split_tokenized(tok_line, max_length=max_small_seq_len)
            current_small_cache += small_tok_lines
            big_tok_lines = text_processor.split_tokenized(tok_line, max_length=max_big_seq_len)
            current_big_cache += big_tok_lines

            if len(current_small_cache) >= 100000:
                for tok_line in current_small_cache:
                    # assuming that every list has same length due to correct padding.
                    small_examples[small_line_num] = torch.LongTensor(tok_line)
                    small_line_num += 1
                    if len(small_examples) >= sentence_small_block_size:
                        with open(os.path.join(small_cache_dir, str(small_file_count) + ".pkl"), "wb") as fw:
                            pickle.dump(small_examples, fw)
                        small_examples, small_file_count = {}, small_file_count + 1
                current_small_cache = []
                print(
                    f"from {ln} actual documents, dumped {small_line_num} small vectors into {small_file_count} files")
            if len(current_big_cache) >= 100000:
                for tok_line in current_big_cache:
                    # assuming that every list has same length due to correct padding.
                    big_examples[big_line_num] = torch.LongTensor(tok_line)
                    big_line_num += 1
                    if len(big_examples) >= sentence_big_block_size:
                        with open(os.path.join(big_cache_dir, str(big_file_count) + ".pkl"), "wb") as fw:
                            pickle.dump(big_examples, fw)
                        big_examples, big_file_count = {}, big_file_count + 1
                current_big_cache = []
                print(f"from {ln} actual documents, dumped {big_line_num} big vectors into {big_file_count} files")

    if len(current_small_cache) > 0:
        for tok_line in current_small_cache:
            small_examples[small_line_num] = torch.LongTensor(tok_line)
            small_line_num += 1
            if len(small_examples) >= sentence_small_block_size:
                with open(os.path.join(small_cache_dir, str(small_file_count) + ".pkl"), "wb") as fw:
                    pickle.dump(small_examples, fw)
                small_examples, small_file_count = {}, small_file_count + 1
        if len(small_examples) >= 0:
            with open(os.path.join(small_cache_dir, str(small_file_count) + ".pkl"), "wb") as fw:
                pickle.dump(small_examples, fw)
            small_examples, small_file_count = {}, small_file_count + 1

        print(f"from {ln} actual documents, dumped {small_line_num} small vectors into {small_file_count} files")

    if len(current_big_cache) > 0:
        for tok_line in current_big_cache:
            big_examples[big_line_num] = torch.LongTensor(tok_line)
            big_line_num += 1
            if len(big_examples) >= sentence_big_block_size:
                with open(os.path.join(big_cache_dir, str(big_file_count) + ".pkl"), "wb") as fw:
                    pickle.dump(big_examples, fw)
                big_examples, big_file_count = {}, big_file_count + 1
        if len(big_examples) >= 0:
            with open(os.path.join(big_cache_dir, str(big_file_count) + ".pkl"), "wb") as fw:
                pickle.dump(big_examples, fw)
            big_examples, big_file_count = {}, big_file_count + 1

        print(f"from {ln} actual documents, dumped {big_line_num} big vectors into {big_file_count} files")

    with open(os.path.join(small_cache_dir, "info.txt"), "w") as fw:
        fw.write(str(sentence_small_block_size) + "\t" + str(small_line_num) + "\t" + str(small_file_count))
    with open(os.path.join(big_cache_dir, "info.txt"), "w") as fw:
        fw.write(str(sentence_big_block_size) + "\t" + str(big_line_num) + "\t" + str(big_file_count))


def get_tokenizer(tokenizer_path: Optional[str] = None, train_path: Optional[str] = None,
                  model_path: Optional[str] = None, vocab_size: Optional[int] = None) -> TextProcessor:
    if tokenizer_path is None:
        print("Training Tokenizer...")
        text_processor = TextProcessor()
        print("Writing raw text...")
        with open(train_path + ".tmp", "w") as wf:
            with open(train_path, "r") as rf:
                for i, line in enumerate(rf):
                    spl = [sen.strip() for sen in line.split("</s>") if len(sen.strip()) > 0]
                    if spl[0].startswith("<"):
                        spl[0] = " ".join(spl[0].strip().split(" ")[1:])
                    wf.write("\n".join(spl))
                    wf.write("\n")
                    if (i + 1) % 100000 == 0:
                        print(i + 1)
        print("Writing raw text done!")

        text_processor.train_tokenizer(paths=[train_path + ".tmp"], vocab_size=vocab_size, to_save_dir=model_path)
        print("Removing temporary file!")
        os.system("rm " + train_path + ".tmp &")
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
    if not os.path.exists(options.model_path):
        os.makedirs(options.model_path)
    tokenizer = get_tokenizer(tokenizer_path=options.tokenizer_path, train_path=options.data_path,
                              model_path=options.model_path, vocab_size=options.vocab_size)
    if not os.path.exists(options.small_cache_path):
        os.makedirs(options.small_cache_path)
    if not os.path.exists(options.large_cache_path):
        os.makedirs(options.large_cache_path)

    print("writing batch of 128/512")
    write(text_processor=tokenizer, small_cache_dir=options.small_cache_path, big_cache_dir=options.large_cache_path,
          max_small_seq_len=128, max_big_seq_len=512,
          txt_file=options.data_path, sentence_small_block_size=4 * options.sentence_block,
          sentence_big_block_size=options.sentence_block)
    print("finished")
