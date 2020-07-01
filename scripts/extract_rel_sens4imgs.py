import json
import marshal
import os
import sys
from itertools import chain

sen_chooser = lambda i, s, sens, r, img: (img["img_path"], img["url"], sens[i]) if s > r else None


def extract_shared_sentences(v):
    sens = list(map(lambda s: s.strip() + " </s>", v["content"].split("</s>")))
    sen_words = list(map(lambda s: set(s.split()[1:-1]), sens))
    return list(chain(*map(lambda image: extract_captions4imgs(image, sen_words, sens), v["images"])))


def extract_captions4imgs(image, sen_words, sens):
    caption = image["caption"]
    caption_words = set(caption.strip().split(" ")[1:-1])
    shared_word_counts = list(map(lambda s: len(s & caption_words), sen_words))
    max_word_count = max(shared_word_counts)
    least_req_count = max(2, max_word_count - 2)
    captions = [(image["img_path"], image["url"], caption)] + list(
        filter(lambda x: x != None,
               map(lambda i, s: sen_chooser(i, s, sens, least_req_count, image), range(len(sens)), shared_word_counts)))
    return captions


if __name__ == "__main__":
    json_file = os.path.abspath(sys.argv[1])
    output_file = os.path.abspath(sys.argv[2])

    assert json_file != output_file
    with open(json_file, "rb") as fp:
        doc_dicts = json.load(fp)
        captions = list(chain(*map(lambda v: extract_shared_sentences(v), doc_dicts)))
        print(len(captions))
        with open(output_file, "wb") as wfp:
            marshal.dump(captions, wfp)
        print("Done!")
