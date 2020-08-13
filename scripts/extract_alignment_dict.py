import os
import sys
from collections import Counter

src_path = os.path.abspath(sys.argv[1])
dst_path = os.path.abspath(sys.argv[2])
alignment_path = os.path.abspath(sys.argv[3])
dict_path = os.path.abspath(sys.argv[4])

src_word_count = Counter()
dst_word_count = Counter()
cooc_count = Counter()

with open(src_path, "r") as sr, open(dst_path, "r") as dr, open(alignment_path, "r") as ar:
    for src, dst, alignment in zip(sr, dr, ar):
        src_words = src.strip().split(" ")
        dst_words = dst.strip().split(" ")

        src_word_count += Counter(src_words)
        dst_word_count += Counter(dst_words)

        for a in alignment.strip().split(" "):
            spl = a.split("-")
            if len(spl) != 2: continue

            si, di = int(spl[0]), int(spl[1])

            cooc_count[src_words[si] + "\t" + dst_words[di]] += 1

with open(dict_path, "w") as writer:
    for word_pair in cooc_count:
        src_word, dst_word = word_pair.split("\t")
        pmi = cooc_count[word_pair] / (src_word_count[src_word] * dst_word_count[dst_word])
        writer.write(word_pair + "\t" + str(pmi) + "\n")
