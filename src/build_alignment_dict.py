from collections import defaultdict
from optparse import OptionParser

from textprocessor import TextProcessor


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--src", dest="src", metavar="FILE", default=None)
    parser.add_option("--dst", dest="dst", metavar="FILE", default=None)
    parser.add_option("--align", dest="align", metavar="FILE", default=None)
    parser.add_option("--output", dest="output", metavar="FILE", default=None)
    parser.add_option("--tok", dest="tok", help="Path to the tokenizer folder", metavar="FILE", default=None)
    (options, args) = parser.parse_args()
    return options


options = get_options()

src_file = options.src
dst_file = options.dst
align_file = options.align
output_file = options.output
tokenizer: TextProcessor = TextProcessor(options.tok)

word_translation = defaultdict(dict)
word_counter = defaultdict(int)

with open(src_file, "r") as sr, open(dst_file, "r") as dr, open(align_file, "r") as ar:
    for i, (src_line, dst_line, align_line) in enumerate(zip(sr, dr, ar)):
        src_words = src_line.strip().split(" ")
        dst_words = dst_line.strip().split(" ")
        alignments = filter(lambda x: x != None,
                            map(lambda a: (int(a.split("-")[0]), int(a.split("-")[1])) if "-" in a else None,
                                align_line.strip().split(" ")))
        for a in alignments:
            src_word, dst_word = tokenizer.token_id(src_words[a[0]]), tokenizer.token_id(dst_words[a[1]])
            if dst_word not in word_translation[src_word]:
                word_translation[src_word][dst_word] = 1
                word_translation[dst_word][src_word] = 1
            else:
                word_translation[src_word][dst_word] += 1
                word_translation[dst_word][src_word] += 1
            word_counter[src_word] += 1
            word_counter[dst_word] += 1
        if (i + 1) % 100000 == 0:
            print(i + 1, len(word_translation), len(word_counter), end="\r")

max_align_len = 0
with open(output_file, "w") as writer:
    for w in word_translation.keys():
        output = [str(w)]
        denom = word_counter[w]
        for t in word_translation[w].keys():
            word_translation[w][t] /= denom
        sort_orders = sorted(word_translation[w].items(), key=lambda x: x[1], reverse=True)[:5]
        for so in sort_orders:
            output.append(str(so[0]))
        max_align_len = max(len(word_translation[w]), max_align_len)
        writer.write(" ".join(output) + "\n")
print(max_align_len)
