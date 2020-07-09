from optparse import OptionParser

import sacrebleu


def get_lm_option_parser():
    parser = OptionParser()
    parser.add_option("--output", dest="output", metavar="FILE", default=None)
    parser.add_option("--gold", dest="gold", metavar="FILE", default=None)
    return parser


(options, args) = get_lm_option_parser().parse_args()

output = open(options.output, "r").read().strip().split("\n")
gold = [open(options.gold, "r").read().strip().split("\n")]

print("Cased BLEU")
bleu = sacrebleu.corpus_bleu(output, gold)
print(bleu)
print(bleu.score)

print("Lowercased BLEU")
bleu = sacrebleu.corpus_bleu(output, gold, lowercase=True)
print(bleu)
print(bleu.score)
