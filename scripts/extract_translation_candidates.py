import json
from collections import defaultdict
from itertools import chain
from optparse import OptionParser

"""
Extracts all images with all sentences (longer than 5 words).
"""
sen_chooser = lambda sens, img: list(map(lambda s: (img, s), sens))
img_sen_collect = lambda image, sens: [(image["img_path"], image["caption"])] + sen_chooser(sens, image["img_path"])
len_condition = lambda words1, words2: True if .8 <= len(words1) / len(words2) <= 1.1 or abs(
    len(words1) - len(words2)) <= 5 else False
img_sen_pair_collect = lambda image, rs, sens, output_image: list(filter(lambda x: x is not None, map(
    lambda s: ((image, s, rs) if output_image else (s, rs)) if len_condition(s.split(" "), rs.split(
        " ")) else None, sens)))


def extract_sentences(v):
    content_spl = v["content"].strip().split(" ")
    lang_id, content = content_spl[0] + " ", " ".join(content_spl[1:])
    sens = list(filter(lambda x: x != None,
                       map(lambda s: lang_id + s.strip() + " </s>" if 256 >= len(s.strip().split(" ")) >= 5 else None,
                           content.split("</s>"))))
    result = list(chain(*map(lambda img: img_sen_collect(img, sens), v["images"])))
    return result


def extract_sentence_pairs(v, ref_images, ref_captions, output_image):
    shared_images = list(filter(lambda x: x is not None,
                                map(lambda img: img["img_path"] if img["img_path"] in ref_images else None,
                                    v["images"])))
    if len(shared_images) == 0:
        return []
    content_spl = v["content"].strip().split(" ")
    lang_id, content = content_spl[0] + " ", " ".join(content_spl[1:])
    sens = list(filter(lambda x: x != None,
                       map(lambda s: lang_id + s.strip() + " </s>" if len(s.strip().split(" ")) >= 5 else None,
                           content.split("</s>"))))
    captions = {f["img_path"]: f["caption"] for f in v["images"]}
    sentence_pairs = list(chain(*map(
        lambda i: chain(*map(lambda ref_sen: img_sen_pair_collect(i, ref_sen, sens + [captions[i]], output_image),
                             ref_captions[i])),
        shared_images)))
    return sentence_pairs


def write(output_file: str, input_file: str, ref_file=None, output_image=False, src_lang="src", dst_lang="dst"):
    with open(ref_file, "rb") as fp:
        ref_doc_dicts = json.load(fp)
        ref_images = set(chain(*map(lambda v: list(map(lambda im: im["img_path"], v["images"])), ref_doc_dicts)))
        ref_captions = list(chain(*map(lambda v: extract_sentences(v), ref_doc_dicts)))
        ref_caption_dict = defaultdict(set)
        for i, s in ref_captions:
            ref_caption_dict[i].add(s)
        print("Reference Captions", len(ref_captions), len(ref_caption_dict))

    with open(input_file, "rb") as fp, open(output_file + "." + src_lang, "w") as src_w, open(
            output_file + "." + dst_lang, "w") as dst_w:
        doc_dicts = json.load(fp)
        for doc_dict in doc_dicts:
            sentence_pairs = extract_sentence_pairs(doc_dict, ref_images, ref_caption_dict, output_image)
            if len(sentence_pairs) == 0:
                continue
            src_w.write("\n".join(list(map(lambda s: s[0], sentence_pairs))))
            dst_w.write("\n".join(list(map(lambda s: s[1], sentence_pairs))))
            src_w.write("\n")
            dst_w.write("\n")
            print(len(sentence_pairs), end="\r")
        print("Done!")


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--file", dest="file", help="Which files to use", metavar="FILE", default=None)
    parser.add_option("--ref", dest="ref", help="Ref files to use for overlap", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_file", help="Output pickle file.", metavar="FILE", default=None)
    parser.add_option("--image", action="store_true", dest="output_image", help="Output image path as well",
                      default=False)
    parser.add_option("--src", dest="src_lang", type="str", default="src")
    parser.add_option("--dst", dest="dst_lang", type="str", default="dst")
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()

    print("Writing batches")
    write(output_file=options.output_file,
          input_file=options.file,
          ref_file=options.ref,
          output_image=options.output_image,
          src_lang=options.src_lang,
          dst_lang=options.dst_lang)
    print("Finished")
