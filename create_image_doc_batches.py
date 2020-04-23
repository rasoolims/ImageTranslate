import json
import os
import pickle
from collections import defaultdict
from optparse import OptionParser

from textprocessor import TextProcessor


def write(text_processor: TextProcessor, output_file: str,
          max_seq_len: int, json_dir: str, files_to_use: str = None):
    relevant_files = None
    if files_to_use is not None:
        relevant_files = {f + ".json" for f in files_to_use.strip().split(",")}

    num_captions, num_docs, max_doc_size = 0, 0, 0
    image_data = defaultdict(list)
    for file in os.listdir(json_dir):
        if not file.endswith(".json"):
            continue
        if relevant_files is not None and file not in relevant_files:
            continue

        print(file)
        with open(os.path.join(json_dir, file), "rb") as fp:
            doc_dicts = json.load(fp)

            for doc in doc_dicts:
                content = doc["content"]
                lang = doc["lang"]
                tok_line = text_processor.tokenize_one_line(content.strip())
                tok_lines = text_processor.split_tokenized(tok_line, max_length=max_seq_len)
                max_doc_size = max(max_doc_size, len(tok_lines))
                num_docs += 1
                num_captions += len(doc["images"])
                for image in doc["images"]:
                    path = image["img_path"]
                    caption = image["caption"]

                    entry = {"caption": caption, "language": lang, "content": tok_lines}
                    image_data[path].append(entry)

            print(len(doc_dicts))
    print(
        "num images %d, docs %d, captions %d, max doc vec %d" % (len(image_data), num_docs, num_captions, max_doc_size))
    with open(output_file, "wb") as fp:
        pickle.dump(image_data, fp)


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--data", dest="data_path", help="Path to the data folder", metavar="FILE", default=None)
    parser.add_option("--files", dest="files_to_use", help="Which files to use", type="str", default=None)
    parser.add_option("--output", dest="output_file", help="Output pickle file.", metavar="FILE", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    tokenizer = TextProcessor(options.tokenizer_path)

    print("writing batches")
    write(text_processor=tokenizer,
          output_file=options.output_file,
          max_seq_len=512,
          json_dir=options.data_path,
          files_to_use=options.files_to_use)
    print("finished")
