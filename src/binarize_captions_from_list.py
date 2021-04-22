import marshal
import os
from optparse import OptionParser

from textprocessor import TextProcessor

id2path = lambda path: "".join(["".join((12 - len(path)) * ["0"]), path, ".jpg"])
caption_format = lambda caption: " ".join(["<en>", caption, "</s>"])
caption_data = lambda annotation: (id2path(str(annotation["image_id"])), caption_format(annotation["caption"]))


def write(text_processor: TextProcessor, output_file: str, input_file: str, max_len: int, sample_size: int, lang):
    eos = "</s>"
    if lang is not None:
        lang = "<" + lang + ">"

    skipped_long_sens = 0
    image_path_dict, unique_images = dict(), dict()

    tok_captions = {}
    image_ids = {}
    with open(input_file, "r") as r:
        for ci, line in enumerate(r):
            try:
                path, caption = line.strip().split("\t")
                if lang is not None and not caption.startswith(lang):
                    caption = " ".join([lang, caption, eos])
                tok_sen = text_processor.tokenize_one_sentence(caption)
                if len(tok_sen) > max_len:
                    skipped_long_sens += 1
                    continue
                if "." not in path:  # Does not have extension; will add jpg.
                    if os.path.exists(path + ".jpg"):
                        path = path + ".jpg"
                    elif os.path.exists(path + ".jpeg"):
                        path = path + ".jpeg"
                    elif os.path.exists(path + ".JPG"):
                        path = path + ".JPG"
                    elif os.path.exists(path + ".png"):
                        path = path + ".png"
                    elif os.path.exists(path + ".PNG"):
                        path = path + ".PNG"
                if path not in image_path_dict:
                    image_id = len(unique_images)
                    unique_images[image_id] = path
                    image_path_dict[path] = image_id
                elif path in image_path_dict:
                    image_id = image_path_dict[path]
                    unique_images[image_id] = path

                caption_id = len(tok_captions)
                tok_captions[caption_id] = tok_sen
                image_ids[caption_id] = image_id

                if (ci + 1) >= sample_size and sample_size > 0:
                    break
            except:
                print(line.strip())

    print("Skipped long sentences:", skipped_long_sens)
    tok_captions_sorted = sorted(tok_captions.items(), key=lambda item: len(item[1]))
    caption_sorted = list(map(lambda e: (image_ids[e[0]], e[1]), tok_captions_sorted))
    print("Longest sentence", len(tok_captions_sorted[-1][1]))
    with open(output_file, "wb") as wfp:
        marshal.dump((unique_images, caption_sorted), wfp)
    print("Dumped", len(caption_sorted), "captions from", len(unique_images), "unique images")


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--file", dest="file", help="Which files to use", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_file", help="Output pickle file.", metavar="FILE", default=None)
    parser.add_option("--lang", dest="lang", type="str", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--max-len", dest="max_len", help="Maximum tokenized caption length", type="int", default=256)
    parser.add_option("--sample", dest="sample_size", type="int", default=-1)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    tokenizer = TextProcessor(options.tokenizer_path)

    print("Writing batches")
    write(text_processor=tokenizer,
          output_file=options.output_file,
          input_file=options.file,
          max_len=options.max_len,
          sample_size=options.sample_size,
          lang=options.lang)
    print("Finished")
