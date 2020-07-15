import json
import marshal
from collections import defaultdict
from itertools import chain
from optparse import OptionParser

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


def write(output_file: str, input_file: str, ref_file=None, output_image=False):
    with open(ref_file, "rb") as fp:
        ref_doc_dicts = json.load(fp)
        ref_images = set(chain(*map(lambda v: list(map(lambda im: im["img_path"], v["images"])), ref_doc_dicts)))
        ref_captions = list(chain(*map(lambda v: extract_sentences(v), ref_doc_dicts)))
        ref_caption_dict = defaultdict(set)
        for i, s in ref_captions:
            ref_caption_dict[i].add(s)
        print("Reference Captions", len(ref_captions), len(ref_caption_dict))

    sen_ids = dict()
    sentences = []
    src2dst_dict = defaultdict(set)
    dst2src_dict = defaultdict(set)
    with open(input_file, "rb") as fp, open(output_file, "wb") as writer:
        doc_dicts = json.load(fp)
        for i, doc_dict in enumerate(doc_dicts):
            sentence_pairs = extract_sentence_pairs(doc_dict, ref_images, ref_caption_dict, output_image)
            if len(sentence_pairs) == 0:
                continue
            for src, dst in sentence_pairs:
                if src not in sen_ids:
                    sen_ids[src] = len(sen_ids)
                    sentences.append(src)
                if dst not in sen_ids:
                    sen_ids[dst] = len(sen_ids)
                    sentences.append(dst)

                src2dst_dict[sen_ids[src]].add(sen_ids[dst])
                dst2src_dict[sen_ids[dst]].add(sen_ids[src])

            print(i, "/", len(doc_dict), end="\r")
        marshal.dump((sen_ids, dict(src2dst_dict), dict(dst2src_dict)), writer)


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--file", dest="file", help="Which files to use", metavar="FILE", default=None)
    parser.add_option("--ref", dest="ref", help="Ref files to use for overlap", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_file", help="Output pickle file.", metavar="FILE", default=None)
    parser.add_option("--image", action="store_true", dest="output_image", help="Output image path as well",
                      default=False)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()

    print("Writing batches")
    write(output_file=options.output_file,
          input_file=options.file,
          ref_file=options.ref,
          output_image=options.output_image)
    print("\nFinished")
