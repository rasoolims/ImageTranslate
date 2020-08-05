import json
from collections import defaultdict
from optparse import OptionParser

import sacrebleu


def get_lm_option_parser():
    parser = OptionParser()
    parser.add_option("--output", dest="output", metavar="FILE", default=None)
    parser.add_option("--gold", dest="gold", metavar="FILE", default=None)
    return parser


(options, args) = get_lm_option_parser().parse_args()

output = dict()
with open(options.output, "r") as r:
    for line in r:
        path, caption = line.strip().split("\t")
        if "/" in path:
            path = path[path.rfind("/") + 1:]
        output[path] = caption

with open(options.gold, "r") as r:
    obj = json.load(r)

annotations = obj["annotations"]

caption_dict = defaultdict(list)
min_len = 100
for annotation in annotations:
    caption = annotation["caption"].strip()
    image_path = str(annotation["image_id"])
    added_zeros = "".join((12 - len(image_path)) * ["0"])
    image_path = "".join([added_zeros, image_path, ".jpg"])
    caption_dict[image_path].append(caption)

print(len(annotations))
for path in caption_dict.keys():
    min_len = min(len(caption_dict[path]), min_len)

sys_out = []
gold = [[], [], [], []]

print(len(output), len(caption_dict))
for path in caption_dict.keys():
    sys_out.append(output[path])
    for i in range(len(gold)):
        gold[i].append(caption_dict[path][i])
print(len(sys_out), len(gold))

print("Cased Detokenized BLEU")
bleu = sacrebleu.corpus_bleu(sys_out, gold)
print(bleu)
print(bleu.score)

print("Cased BLEU")
bleu = sacrebleu.corpus_bleu(sys_out, gold, tokenize="intl")
print(bleu)
print(bleu.score)

print("Lowercased BLEU")
bleu = sacrebleu.corpus_bleu(sys_out, gold, lowercase=True, tokenize="intl")
print(bleu)
print(bleu.score)
