import json
import random
from optparse import OptionParser


def get_option_parser():
    parser = OptionParser()
    parser.add_option("--l1j", dest="l1_json", help="Json path for the first language", metavar="FILE", default=None)
    parser.add_option("--l2j", dest="l2_json", help="Json path for the second language", metavar="FILE", default=None)
    parser.add_option("--l1r", dest="l1_raw", help="Raw txt for the first language", metavar="FILE", default=None)
    parser.add_option("--l2r", dest="l2_raw", help="Raw txt for the second language", metavar="FILE", default=None)
    parser.add_option("--o1", dest="o1", help="Output txt for the first language", metavar="FILE", default=None)
    parser.add_option("--o2", dest="o2", help="Output txt for the second language", metavar="FILE", default=None)
    parser.add_option("--min_doc", dest="min_doc", help="Min number of doctences", type=int, default=1000000)
    return parser


parser = get_option_parser()
(options, args) = parser.parse_args()

docs1 = set()
with open(options.l1_json, "rb") as fp:
    contents = json.load(fp)
    for i, content in enumerate(contents):
        docs1.add(content["content"].strip())
        docs1 |= set(map(lambda img: img["caption"], content["images"]))
        print(i, end="\r")

print(len(docs1), "docs in", options.l1_json)

docs2 = set()
with open(options.l2_json, "rb") as fp:
    contents = json.load(fp)
    for i, content in enumerate(contents):
        docs2.add(content["content"].strip())
        docs2 |= set(map(lambda img: img["caption"], content["images"]))
        print(i, end="\r")
print(len(docs2), "docs in", options.l2_json)

raw_doc1 = set()
with open(options.l1_raw, "r") as reader:
    for i, line in enumerate(reader):
        if line.strip() not in docs1:
            raw_doc1.add(line.strip())
        print(i, end="\r")

print(len(raw_doc1), "docs doctences in", options.l1_raw)

raw_doc2 = set()
with open(options.l2_raw, "r") as reader:
    for i, line in enumerate(reader):
        if line.strip() not in docs2:
            raw_doc2.add(line.strip())
        print(i, end="\r")

print(len(raw_doc2), "docs doctences in", options.l2_raw)

docs1 = list(docs1)
docs2 = list(docs2)
raw_doc1 = list(raw_doc1)
raw_doc2 = list(raw_doc2)

min_doc = min(len(docs1), len(docs2))
max_doc = max(len(docs1), len(docs2))

min_needed = min(max_doc, options.min_doc)

l1_needed = min(len(raw_doc1), max(0, min_needed - len(docs1)))
l2_needed = min(len(raw_doc2), max(0, min_needed - len(docs2)))

print(l1_needed, l2_needed)
if l1_needed > 0:
    random.shuffle(raw_doc1)
    docs1 += raw_doc1[:l1_needed]

if l2_needed > 0:
    random.shuffle(raw_doc2)
    docs2 += raw_doc2[:l2_needed]

with open(options.o1, "w") as w:
    w.write("\n".join(docs1))

with open(options.o2, "w") as w:
    w.write("\n".join(docs2))

print("Done!")
