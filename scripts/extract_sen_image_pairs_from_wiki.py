import json
import marshal
import os
import sys
from itertools import chain

"""
Extracts all images with all sentences (longer than 5 words).
"""
sen_chooser = lambda sens, img: list(map(lambda s: (img, s), sens))
img_sen_collect = lambda image, sens: [(image["img_path"], image["caption"])] + sen_chooser(sens, image["img_path"])


def extract_sentences(v):
    content_spl = v["content"].strip().split(" ")
    lang_id, content = content_spl[0] + " ", " ".join(content_spl[1:])
    sens = list(filter(lambda x: x != None,
                       map(lambda s: lang_id + s.strip() + " </s>" if len(s.strip().split(" ")) >= 5 else None,
                           content.split("</s>"))))
    result = list(chain(*map(lambda img: img_sen_collect(img, sens), v["images"])))
    return result


if __name__ == "__main__":
    json_file = os.path.abspath(sys.argv[1])
    output_file = os.path.abspath(sys.argv[2])

    assert json_file != output_file
    with open(json_file, "rb") as fp:
        doc_dicts = json.load(fp)
        num_captions = sum(list(map(lambda v: len(v["images"]), doc_dicts)))
        captions = list(chain(*map(lambda v: extract_sentences(v), doc_dicts)))
        print(num_captions, len(captions))
        with open(output_file, "wb") as wfp:
            marshal.dump(captions, wfp)
        print("Done!")
