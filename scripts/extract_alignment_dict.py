import os
import sys
from collections import Counter

src_path = os.path.abspath(sys.argv[1])
dst_path = os.path.abspath(sys.argv[2])
alignment_path = os.path.abspath(sys.argv[3])
min_cooc = int(sys.argv[4])
dict_path = os.path.abspath(sys.argv[5])

src_word_count = Counter()
dst_word_count = Counter()
cooc_count = Counter()

alignment_counter = lambda alignment, src_words, dst_words: Counter(
    map(lambda a: src_words[int(a[0])] + "\t" + dst_words[int(a[1])], alignment))

with open(src_path, "r") as sr, open(dst_path, "r") as dr, open(alignment_path, "r") as ar:
    for i, (src, dst, alignment) in enumerate(zip(sr, dr, ar)):
        src_words = src.strip().split(" ")
        dst_words = dst.strip().split(" ")

        src_word_count += Counter(src_words)
        dst_word_count += Counter(dst_words)
        try:
            alignments = filter(lambda x: len(x) == 2, map(lambda a: a.split("-"), alignment.strip().split(" ")))
            cooc_count += alignment_counter(alignments, src_words, dst_words)
        except:
            pass
        print(i, end="\r")

print("\nDict processing")
written = 0
with open(dict_path, "w") as writer:
    for i, word_pair in enumerate(cooc_count):
        src_word, dst_word = word_pair.split("\t")
        if cooc_count[word_pair] < min_cooc:
            continue
        pmi = cooc_count[word_pair] / (src_word_count[src_word] * dst_word_count[dst_word])
        writer.write(word_pair + "\t" + str(pmi) + "\n")
        written += 1
        print(written, "/", i, end="\r")
print("\nDone!")
