import json
import random
from optparse import OptionParser


def extract_sentences(line, min_len):
    line = line.strip()
    if len(line) == 0:
        return set()

    sens = line.split("</s>")
    sen_split = sens[0].strip().split(" ")
    sens[0] = " ".join(sen_split[1:])
    lang = sen_split[0]
    len_condition = lambda s: len(s.strip()) > 0 and len(s.strip().split(" ")) > min_len
    return set(filter(lambda x: x is not None,
                      map(lambda s: " ".join([lang, s.strip(), "</s>"]) if len_condition(s) else None, sens)))


def get_option_parser():
    parser = OptionParser()
    parser.add_option("--l1j", dest="l1_json", help="Json path for the first language", metavar="FILE", default=None)
    parser.add_option("--l2j", dest="l2_json", help="Json path for the second language", metavar="FILE", default=None)
    parser.add_option("--l1r", dest="l1_raw", help="Raw txt for the first language", metavar="FILE", default=None)
    parser.add_option("--l2r", dest="l2_raw", help="Raw txt for the second language", metavar="FILE", default=None)
    parser.add_option("--o1", dest="o1", help="Output txt for the first language", metavar="FILE", default=None)
    parser.add_option("--o2", dest="o2", help="Output txt for the second language", metavar="FILE", default=None)
    parser.add_option("--min_sen", dest="min_sen", help="Min number of sentences", type=int, default=1000000)
    parser.add_option("--min_len", dest="min_len", help="Min length of raw sentences", type=int, default=5)
    return parser


parser = get_option_parser()
(options, args) = parser.parse_args()

sens1 = set()
with open(options.l1_json, "rb") as fp:
    contents = json.load(fp)
    for i, content in enumerate(contents):
        sens1 |= extract_sentences(content["content"], 0)
        sens1 |= set(map(lambda img: img["caption"], content["images"]))
        print(i, end="\r")

print(len(sens1), "sens in", options.l1_json)

sens2 = set()
with open(options.l2_json, "rb") as fp:
    contents = json.load(fp)
    for i, content in enumerate(contents):
        sens2 |= extract_sentences(content["content"], 0)
        sens2 |= set(map(lambda img: img["caption"], content["images"]))
        print(i, end="\r")
print(len(sens2), "sens in", options.l2_json)

raw_sen1 = set()
with open(options.l1_raw, "r") as reader:
    for i, line in enumerate(reader):
        raw_sen1 |= extract_sentences(line, options.min_len)
        print(i, end="\r")

print(len(raw_sen1), "raw sentences in", options.l1_raw)

raw_sen2 = set()
with open(options.l1_raw, "r") as reader:
    for i, line in enumerate(reader):
        raw_sen2 |= extract_sentences(line, options.min_len)
        print(i, end="\r")

print(len(raw_sen2), "raw sentences in", options.l2_raw)

sens1 = list(sens1)
sens2 = list(sens2)
raw_sen1 = list(raw_sen1)
raw_sen2 = list(raw_sen2)
random.shuffle(raw_sen1)
random.shuffle(raw_sen2)

min_sen = min(len(sens1), len(sens2))
max_sen = max(len(sens1), len(sens2))

min_needed = min(max_sen, options.min_sen)

l1_needed = min(len(raw_sen1), max(0, min_needed - len(sens1)))
l2_needed = min(len(raw_sen2), max(0, min_needed - len(sens2)))

sens1 += raw_sen1[:l1_needed]
sens2 += raw_sen2[:l2_needed]

with open(options.o1, "w") as w:
    w.write("\n".join(sens1))

with open(options.o2, "w") as w:
    w.write("\n".join(sens2))

print("Done!")