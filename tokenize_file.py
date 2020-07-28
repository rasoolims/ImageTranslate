from optparse import OptionParser

from textprocessor import TextProcessor


def write(text_processor: TextProcessor, output_file: str, txt_file: str, output_txt: bool = False):
    with open(txt_file, "r") as fp, open(output_file, "w") as writer:
        for line in fp:
            if len(line.strip()) == 0 or len(line.strip()) == 0: continue
            tok_line = text_processor.tokenize_one_line(line.strip(), ignore_middle_eos=True)

            if output_txt:
                tokenized = [text_processor.id2token(tok) for tok in tok_line][1:-1]
                tokenized = list(map(lambda tok: tok if tok != "<unk>" else "unk", tokenized))
            else:
                tokenized = [str(tok) for tok in tok_line]
            writer.write(" ".join(tokenized) + "\n")


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

    print("writing")
    write(text_processor=tokenizer, output_file=options.output_path, txt_file=options.data_path,
          output_txt=options.output_text)
    print("finished")
