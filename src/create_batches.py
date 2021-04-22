import marshal
import os
from optparse import OptionParser
from typing import Optional

from textprocessor import TextProcessor


def write(text_processor: TextProcessor, cache_dir: str,
          seq_len: int, txt_file: str, sen_block_size: int = 10000):
    sen_block_size = sen_block_size
    current_cache, cur_cache_langs = [], []
    examples = {}
    line_num, file_count = 0, 0
    text_processor.max_len = seq_len

    with open(txt_file, "r") as fp:
        for ln, line in enumerate(fp):
            if len(line.strip()) == 0: continue
            tok_lines = text_processor.tokenize_lines(line.strip(), blind_split=True, split_len=seq_len)
            current_cache += list(tok_lines)
            cur_cache_langs += [text_processor.languages[text_processor.id2token(tok_lines[0, 0])]] * tok_lines.shape[0]

            if len(current_cache) >= 100000:
                for tok_line, lang in zip(current_cache, cur_cache_langs):
                    # assuming that every list has same length due to correct padding.
                    examples[line_num] = (tok_line.tolist(), lang)
                    line_num += 1
                    if len(examples) >= sen_block_size:
                        with open(os.path.join(cache_dir, str(file_count) + ".pkl"), "wb") as fw:
                            marshal.dump(examples, fw)
                        examples, file_count = {}, file_count + 1
                current_cache, cur_cache_langs = [], []
                print(f"from {ln} actual documents, dumped {line_num} big vectors into {file_count} files")

    if len(current_cache) > 0:
        for tok_line, lang in zip(current_cache, cur_cache_langs):
            # assuming that every list has same length due to correct padding.
            examples[line_num] = (tok_line.tolist(), lang)
            line_num += 1
            if len(examples) >= sen_block_size:
                with open(os.path.join(cache_dir, str(file_count) + ".pkl"), "wb") as fw:
                    marshal.dump(examples, fw)
                examples, file_count = {}, file_count + 1
        if len(examples) >= 0:
            with open(os.path.join(cache_dir, str(file_count) + ".pkl"), "wb") as fw:
                marshal.dump(examples, fw)
            examples, file_count = {}, file_count + 1

        print(f"from {ln} actual documents, dumped {line_num} big vectors into {file_count} files")

    with open(os.path.join(cache_dir, "info.txt"), "w") as fw:
        fw.write(str(sen_block_size) + "\t" + str(line_num) + "\t" + str(file_count))


def get_tokenizer(tokenizer_path: Optional[str] = None, train_path: Optional[str] = None,
                  model_path: Optional[str] = None, vocab_size: Optional[int] = None) -> TextProcessor:
    if tokenizer_path is None:
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        print("Training Tokenizer...")
        text_processor = TextProcessor()
        print("Writing raw text...")
        languages = set()
        with open(train_path + ".tmp", "w") as wf:
            with open(train_path, "r") as rf:
                for i, line in enumerate(rf):
                    spl = [sen.strip() for sen in line.split("</s>") if len(sen.strip()) > 0]
                    if len(spl) == 0: continue
                    if spl[0].startswith("<"):
                        sen_split = spl[0].strip().split(" ")
                        spl[0] = " ".join(sen_split[1:])
                        languages.add(sen_split[0])
                    wf.write("\n".join(spl))
                    wf.write("\n")
                    if ((i + 1) % 1000 == 0):
                        print(i + 1, "\r", end="")
        print("Writing raw text done!")

        text_processor.train_tokenizer(paths=[train_path + ".tmp"], vocab_size=vocab_size, to_save_dir=model_path,
                                       languages={l: i for i, l in enumerate(sorted(languages))})
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
    parser.add_option("--cache", dest="cache_path",
                      help="Path to the data marshal files for data with sequence length", metavar="FILE",
                      default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--block", dest="sentence_block", help="Sentence block size", type="int", default=10000)
    parser.add_option("--len", dest="seq_len", help="Maximum sequence length", type="int", default=512)
    parser.add_option("--vocab_size", dest="vocab_size", help="Vocabulary size", type="int", default=30000)
    parser.add_option("--model", dest="model_path", help="Directory path to save the best model", metavar="FILE",
                      default=None)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()

    tokenizer = get_tokenizer(tokenizer_path=options.tokenizer_path, train_path=options.data_path,
                              model_path=options.model_path, vocab_size=options.vocab_size)
    if not os.path.exists(options.cache_path):
        os.makedirs(options.cache_path)

    print("writing batches")
    write(text_processor=tokenizer, cache_dir=options.cache_path,
          seq_len=options.seq_len,
          txt_file=options.data_path,
          sen_block_size=options.sentence_block)
    print("finished")
