import os
import sys
from collections import Counter

fast_align_path = os.path.abspath(sys.argv[1])
alignment_path = os.path.abspath(sys.argv[2])
min_cooc = int(sys.argv[3])
dict_path = os.path.abspath(sys.argv[4])

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
            print(repr(err))
            print(alignment.strip().split(" "))
            print(src)
            print(dst)
        print(i, end="\r")

cooc_count = Counter(coocs)
pair_dict = sorted(cooc_count.items(), key=lambda x: x[1], reverse=True)
covered = set()

print("\nDict processing")
written = 0
with open(dict_path, "w") as writer:
    for i, (word_pair, count) in enumerate(pair_dict):
        src_word, dst_word = word_pair.split("\t")
        if src_word.lower().strip() == dst_word.lower().strip():
            continue
        if src_word in covered:
            continue
        if count < min_cooc:
            continue
        covered.add(src_word)
        writer.write(word_pair + "\n")

        upper_cased = src_word[0].upper() + src_word[1:] + "\t" +  dst_word[0].upper() + dst_word[1:]
        written += 1
        if upper_cased != word_pair:
            writer.write(upper_cased + "\n")
            written += 1
        print(written, "/", i, end="\r")
print("\nDone!")
