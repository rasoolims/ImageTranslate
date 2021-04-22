import json
import marshal
from optparse import OptionParser

from textprocessor import TextProcessor

id2path = lambda path: "".join(["".join((12 - len(path)) * ["0"]), path, ".jpg"])
caption_format = lambda caption: " ".join(["<en>", caption, "</s>"])
caption_data = lambda annotation: (id2path(str(annotation["image_id"])), caption_format(annotation["caption"]))


def write(text_processor: TextProcessor, output_file: str, input_file: str, max_len: int, sample_size: int):
    with open(input_file, "r") as r:
        obj = json.load(r)

    annotations = obj["annotations"]
    captions = list(map(lambda annotation: caption_data(annotation), annotations))
    print(len(captions))

    skipped_long_sens = 0
    image_path_dict, unique_images = dict(), dict()

    tok_captions = {}
    image_ids = {}
    for ci, c in enumerate(captions):
        if ci % 1000 == 0:
            print(ci, "/", len(captions), "->", len(tok_captions), len(unique_images), end="\r")
        tok_sen = text_processor.tokenize_one_sentence(c[1])
        if len(tok_sen) > max_len:
            skipped_long_sens += 1
            continue

        path = c[0]
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

    print("Skipped long sentences:", skipped_long_sens, "from", len(captions))
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
          sample_size=options.sample_size)
    print("Finished")
