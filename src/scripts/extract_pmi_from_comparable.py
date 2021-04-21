import json
import math
from collections import defaultdict, Counter
from itertools import chain
from optparse import OptionParser

sen_chooser = lambda sens, img: list(map(lambda s: (img, s), sens))
img_sen_collect = lambda image, sens: [(image["img_path"], image["caption"])] + sen_chooser(sens, image["img_path"])
len_condition = lambda words1, words2: True if .9 <= len(words1) / len(words2) <= 1.1 or abs(
    len(words1) - len(words2)) <= 3 else False
img_sen_pair_collect = lambda image, rs, sens: list(filter(lambda x: x is not None, map(
    lambda s: (s, rs) if len_condition(s.split(" "), rs.split(" ")) else None, sens)))


def extract_sentences(v):
    content_spl = v["content"].strip().split(" ")
    lang_id, content = content_spl[0] + " ", " ".join(content_spl[1:])
    sens = list(filter(lambda x: x != None,
                       map(lambda s: lang_id + s.strip() + " </s>" if 256 >= len(s.strip().split(" ")) >= 5 else None,
                           content.split("</s>"))))
    result = list(chain(*map(lambda img: img_sen_collect(img, sens), v["images"])))
    return result


def extract_sentence_pairs(v, ref_images, ref_captions):
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
        lambda i: chain(*map(lambda ref_sen: img_sen_pair_collect(i, ref_sen, sens + [captions[i]]),
                             ref_captions[i])),
        shared_images)))
    return sentence_pairs


def write(output_file: str, input_file: str, ref_file=None):
    with open(ref_file, "rb") as fp:
        ref_doc_dicts = json.load(fp)
        ref_images = set(chain(*map(lambda v: list(map(lambda im: im["img_path"], v["images"])), ref_doc_dicts)))
        ref_captions = list(chain(*map(lambda v: extract_sentences(v), ref_doc_dicts)))
        ref_caption_dict = defaultdict(set)
        for i, s in ref_captions:
            ref_caption_dict[i].add(s)
        print("Reference Captions", len(ref_captions), len(ref_caption_dict))

    src_word_counts = Counter()
    dst_word_counts = Counter()
    cooc = defaultdict(Counter)
    cooc_sum = Counter()
    with open(input_file, "rb") as fp, open(output_file, "w") as writer:
        doc_dicts = json.load(fp)
        for i, doc_dict in enumerate(doc_dicts):
            sentence_pairs = extract_sentence_pairs(doc_dict, ref_images, ref_caption_dict)

            if len(sentence_pairs) == 0:
                continue

            for (src, dst) in sentence_pairs:
                src_counts = Counter(src.strip().split(" ")[1:-1])
                dst_counts = Counter(dst.strip().split(" ")[1:-1])
                src_word_counts += src_counts
                dst_word_counts += dst_counts

                denom = sum(dst_counts.values())
                for src_word in src_counts:
                    for dst_word in dst_counts:
                        v = src_counts[src_word] * dst_counts[dst_word] / denom
                        cooc[src_word][dst_word] += v
                        cooc_sum[src_word] += v

            print(i, "/", len(doc_dicts), end="\r")
        sum_src_count = math.log(sum(src_word_counts.values()))
        sum_dst_count = math.log(sum(dst_word_counts.values()))
        print("\nFinished counting")
        pmis = Counter()
        for si, src_word in enumerate(cooc):
            p_x = math.log(src_word_counts[src_word]) - sum_src_count
            denom = math.log(cooc_sum[src_word])
            for dst_word in cooc[src_word]:
                p_y = math.log(dst_word_counts[dst_word]) - sum_dst_count
                p_x_y = math.log(cooc[src_word][dst_word]) - denom
                pmis[src_word + "\t" + dst_word] = p_x_y - (p_x + p_y)
            print(si, "/", len(cooc), end="\r")
        most_probable = pmis.most_common(1000000)
        output = "\n".join(list(map(lambda m: m[0] + "\t" + str(m[1]), most_probable)))
        writer.write(output)
        print("\nFinished PMI calculations!")


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--file", dest="file", help="Which files to use", metavar="FILE", default=None)
    parser.add_option("--ref", dest="ref", help="Ref files to use for overlap", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_file", help="Output pickle file.", metavar="FILE", default=None)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()

    print("Writing batches")
    write(output_file=options.output_file,
          input_file=options.file,
          ref_file=options.ref)
    print("\nFinished")
