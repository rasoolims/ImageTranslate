import pickle
from optparse import OptionParser

import torch

from textprocessor import TextProcessor


def write(text_processor: TextProcessor, output_file: str, src_txt_file: str, dst_txt_file: str = None):
    examples = {}
    line_num = 0

    lens = {}
    if dst_txt_file is not None:
        with open(src_txt_file, "r") as s_fp, open(dst_txt_file, "r") as d_fp:
            for src_line, dst_line in zip(s_fp, d_fp):
                if len(src_line.strip()) == 0 or len(dst_line.strip()) == 0: continue
                src_tok_line = text_processor.tokenize_one_sentence(src_line.strip())
                src_lang = text_processor.languages[text_processor.id2token(src_tok_line[0])]
                dst_tok_line = text_processor.tokenize_one_sentence(dst_line.strip())
                dst_lang = text_processor.languages[text_processor.id2token(dst_tok_line[0])]
                examples[line_num] = (
                    torch.LongTensor(src_tok_line), torch.LongTensor(dst_tok_line), src_lang, dst_lang)
                lens[line_num] = len(dst_tok_line)
                line_num += 1

        print("Sorting")
        sorted_lens = sorted(lens.items(), key=lambda item: item[1])
        sorted_examples = []
        print("Sorted examples")
        for len_item in sorted_lens:
            line_num = len(sorted_examples)
            sorted_examples.append(examples[len_item[0]])

        print("Dumping")
        with open(output_file, "wb") as fw:
            pickle.dump(sorted_examples, fw)

    else:
        part_num = 0
        # Used for MASS training where we only have source sentences.
        with open(src_txt_file, "r") as s_fp:
            for src_line in s_fp:
                if len(src_line.strip()) == 0: continue
                src_tok_line = text_processor.tokenize_one_sentence(src_line.strip())
                src_lang = text_processor.languages[text_processor.id2token(src_tok_line[0])]
                examples[line_num] = (torch.LongTensor(src_tok_line), src_lang)
                lens[line_num] = len(src_tok_line)
                line_num += 1
                if line_num % 1000 == 0:
                    print(line_num, "\r", end="")

        print("\nSorting")
        sorted_lens = sorted(lens.items(), key=lambda item: item[1])
        sorted_examples = []
        print("Sorted examples")
        for len_item in sorted_lens:
            line_num = len(sorted_examples)
            sorted_examples.append(examples[len_item[0]])

            if len(sorted_examples) >= 4000000:
                print("Dumping")
                with open(output_file + "." + str(part_num), "wb") as fw:
                    pickle.dump(sorted_examples, fw)
                sorted_examples = []
                part_num += 1

        if len(sorted_examples) > 0:
            with open(output_file + "." + str(part_num), "wb") as fw:
                pickle.dump(sorted_examples, fw)

    print(f"Dumped {line_num + 1} small vectors!")


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--src", dest="src_data_path", help="Path to the source txt file", metavar="FILE", default=None)
    parser.add_option("--dst", dest="dst_data_path", help="Path to the target txt file", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_path", help="Output pickle file ", metavar="FILE", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    tokenizer = TextProcessor(options.tokenizer_path)

    print("writing batch")
    write(text_processor=tokenizer, output_file=options.output_path, src_txt_file=options.src_data_path,
          dst_txt_file=options.dst_data_path)
    print("finished")
