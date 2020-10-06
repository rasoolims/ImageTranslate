import os
import sys
from collections import Counter

fast_align_path = os.path.abspath(sys.argv[1])
alignment_path = os.path.abspath(sys.argv[2])
dict_path = os.path.abspath(sys.argv[3])

coocs = []
alignment_counter = lambda alignment, src_words, dst_words: list(
    map(lambda a: src_words[int(a[0])] + "\t" + dst_words[int(a[1])], alignment))

all_src_words, all_dst_words = [], []
with open(fast_align_path, "r") as dr, open(alignment_path, "r") as ar:
    for i, (src2dst, alignment) in enumerate(zip(dr, ar)):
        src, dst = src2dst.strip().split(" ||| ")
        src_words = src.strip().split(" ")
        dst_words = dst.strip().split(" ")
        try:
            alignments = filter(lambda x: len(x) == 2, map(lambda a: a.split("-"), alignment.strip().split(" ")))
            coocs += alignment_counter(alignments, src_words, dst_words)
        except Exception as err:
            pass
        print(i, end="\r")

cooc_count = Counter(coocs)
src2dst_dict = dict()
for word_pair in cooc_count.keys():
    count = cooc_count[word_pair]
    src_word, dst_word = word_pair.split("\t")
    if src_word not in src2dst_dict.keys():
        src2dst_dict[src_word] = (dst_word, count)
    elif src2dst_dict[src_word][1] < count:
        src2dst_dict[src_word] = (dst_word, count)

pair_dict = sorted(cooc_count.items(), key=lambda x: x[1], reverse=True)

print("\nDict processing")
written = 0
with open(dict_path, "w") as writer:
    for src_word in src2dst_dict.keys():
        dst_word = src2dst_dict[src_word][0]
        try:
            if src_word.lower().strip() == dst_word.lower().strip():
                continue
            word_pair = src_word + " ||| " + dst_word
            writer.write(word_pair + "\n")

            upper_cased = src_word[0].upper() + src_word[1:] + " ||| " + dst_word[0].upper() + dst_word[1:]
            written += 1
            if upper_cased != word_pair:
                writer.write(upper_cased + "\n")
                written += 1
            print(written, "/", i, end="\r")
        except:
            pass
print("\nDone!")
