import os
import pickle
from optparse import OptionParser

import torch

from textprocessor import TextProcessor


def write(text_processor: TextProcessor, cache_dir: str, src_txt_file: str, dst_txt_file: str, block_size: int = 40000):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    block_size = block_size
    current_src_cache, current_dst_cache = [], []
    examples = {}
    line_num, file_count = 0, 0

    with open(src_txt_file, "r") as s_fp, open(dst_txt_file, "r") as d_fp:
        for src_line, dst_line in zip(s_fp, d_fp):
            if len(src_line.strip()) == 0 or len(dst_line.strip()) == 0: continue
            current_src_cache += [text_processor.tokenize_one_line(src_line.strip())]
            current_dst_cache += [text_processor.tokenize_one_line(dst_line.strip())]

            if len(current_src_cache) >= 100000:
                for src_tok_line, dst_tok_line in zip(current_src_cache, current_dst_cache):
                    # assuming that every list has same length due to correct padding.
                    examples[line_num] = (torch.LongTensor(src_tok_line), torch.LongTensor(dst_tok_line))
                    line_num += 1
                    if len(examples) >= block_size:
                        with open(os.path.join(cache_dir, "mt." + str(file_count) + ".pkl"), "wb") as fw:
                            pickle.dump(examples, fw)
                        examples, file_count = {}, file_count + 1
                current_src_cache, current_dst_cache = [], []
                print(
                    f"Dumped {line_num} small vectors into {file_count} files")

    if len(current_src_cache) > 0:
        for src_tok_line, dst_tok_line in zip(current_src_cache, current_dst_cache):
            # assuming that every list has same length due to correct padding.
            examples[line_num] = (torch.LongTensor(src_tok_line), torch.LongTensor(dst_tok_line))
            line_num += 1
            if len(examples) >= block_size:
                with open(os.path.join(cache_dir, str(file_count) + ".pkl"), "wb") as fw:
                    pickle.dump(examples, fw)
                examples, file_count = {}, file_count + 1

        if len(examples) >= 0:
            with open(os.path.join(cache_dir, "mt." + str(file_count) + ".pkl"), "wb") as fw:
                pickle.dump(examples, fw)
            examples, file_count = {}, file_count + 1

        print(f"Dumped {line_num} small vectors into {file_count} files")

    with open(os.path.join(cache_dir, "mt.info.txt"), "w") as fw:
        fw.write(str(block_size) + "\t" + str(line_num) + "\t" + str(file_count))


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--src", dest="src_data_path", help="Path to the source txt file", metavar="FILE", default=None)
    parser.add_option("--dst", dest="dst_data_path", help="Path to the target txt file", metavar="FILE", default=None)
    parser.add_option("--cache", dest="cache_path",
                      help="Path to the data pickle files for data with large sequence length", metavar="FILE",
                      default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--block", dest="sentence_block", help="Sentence block size", type="int", default=10000)
    parser.add_option("--model", dest="model_path", help="Directory path to save the best model", metavar="FILE",
                      default=None)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    tokenizer = TextProcessor(options.tokenizer_path)

    if not os.path.exists(options.cache_path):
        os.makedirs(options.cache_path)

    print("writing batch")
    write(text_processor=tokenizer, cache_dir=options.cache_path, src_txt_file=options.src_data_path,
          dst_txt_file=options.dst_data_path, block_size=4 * options.sentence_block)
    print("finished")
