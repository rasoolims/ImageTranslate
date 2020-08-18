import os
from optparse import OptionParser
from typing import Optional

from textprocessor import TextProcessor


def get_tokenizer(train_path: Optional[str] = None,
                  model_path: Optional[str] = None, vocab_size: Optional[int] = None) -> TextProcessor:
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
                if (i + 1) % 10000000:
                    print(i + 1, "\r", end="")
    print("Writing raw text done!")

    print(" ".join(languages))
    text_processor.train_tokenizer(paths=[train_path + ".tmp"], vocab_size=vocab_size, to_save_dir=model_path,
                                   languages={l: i for i, l in enumerate(sorted(languages))})
    print("Removing temporary file!")
    os.system("rm " + train_path + ".tmp &")
    print("done!")


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--data", dest="data_path", help="Path to the data folder", metavar="FILE", default=None)
    parser.add_option("--vocab_size", dest="vocab_size", help="Vocabulary size", type="int", default=30000)
    parser.add_option("--model", dest="model_path", help="Directory path to save the best model", metavar="FILE",
                      default=None)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    tokenizer = get_tokenizer(train_path=options.data_path,
                              model_path=options.model_path, vocab_size=options.vocab_size)
