import pickle
from optparse import OptionParser

import torch

from textprocessor import TextProcessor


def write(text_processor: TextProcessor, output_file: str, txt_file: str, output_txt: bool = False):
    examples = {}

    with open(txt_file, "r") as fp:
        for line in fp:
            if len(line.strip()) == 0 or len(dst_line.strip()) == 0: continue
            src_tok_line = text_processor.tokenize_one_line(line.strip(), ignore_middle_eos=True)
            dst_tok_line = text_processor.tokenize_one_line(dst_line.strip(), ignore_middle_eos=True)
            examples[line_num] = (torch.LongTensor(src_tok_line), torch.LongTensor(dst_tok_line))
            lens[line_num] = len(src_tok_line)
            line_num += 1

    sorted_lens = sorted(lens.items(), key=lambda item: item[1])
    sorted_examples = []
    for len_item in sorted_lens:
        line_num = len(sorted_examples)
        sorted_examples.append(examples[len_item[0]])

    with open(output_file, "wb") as fw:
        pickle.dump(sorted_examples, fw)

    print(f"Dumped {line_num + 1} small vectors!")


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--input", dest="data_path", help="Path to the source txt file", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_path", help="Output pickle file ", metavar="FILE", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--txt", action="store_true", dest="output_text",
                      help="Output tokenized in text format; default: number format", default=False)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    tokenizer = TextProcessor(options.tokenizer_path)

    print("writing batch")
    write(text_processor=tokenizer, output_file=options.output_path, txt_file=options.data_path,
          dst_txt_file=options.dst_data_path)
    print("finished")
