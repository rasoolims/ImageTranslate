import os
import sys
from collections import Counter

src_path = os.path.abspath(sys.argv[1])
dst_path = os.path.abspath(sys.argv[2])
alignment_path = os.path.abspath(sys.argv[3])
min_cooc = int(sys.argv[4])
dict_path = os.path.abspath(sys.argv[5])

coocs = []
alignment_counter = lambda alignment, src_words, dst_words: list(
    map(lambda a: src_words[int(a[0])] + "\t" + dst_words[int(a[1])], alignment))

all_src_words, all_dst_words = [], []
with open(src_path, "r") as sr, open(dst_path, "r") as dr, open(alignment_path, "r") as ar:
    for i, (src, dst, alignment) in enumerate(zip(sr, dr, ar)):
        src_words = src.strip().split(" ")
        dst_words = dst.strip().split(" ")
        try:
            alignments = filter(lambda x: len(x) == 2, map(lambda a: a.split("-"), alignment.strip().split(" ")))
            coocs += alignment_counter(alignments, src_words, dst_words)
        except:
            pass
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
        written += 1
        print(written, "/", i, end="\r")
print("\nDone!")
