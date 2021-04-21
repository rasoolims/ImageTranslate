import os
import sys

len_condition = lambda words1, words2: True if .9 <= len(words1) / len(words2) <= 1.1 or abs(
    len(words1) - len(words2)) <= 5 and len(words1) >= 5 and len(words2) >= 5 else False

src_path = os.path.abspath(sys.argv[1])
dst_path = os.path.abspath(sys.argv[2])
alignment_path = os.path.abspath(sys.argv[3])
min_density = float(sys.argv[4])
output_path = os.path.abspath(sys.argv[5])

written = 0

with open(src_path, "r") as sr, open(dst_path, "r") as dr, open(alignment_path, "r") as ar, open(output_path, "w") as w:
    for i, (src, dst, alignment) in enumerate(zip(sr, dr, ar)):
        src_words = src.strip().split(" ")
        dst_words = dst.strip().split(" ")
        alignments = alignment.strip().split(" ")

        density = len(alignments) / max(len(src_words), len(dst_words))
        if density >= min_density and len(src_words) >= 5 and len(dst_words) >= 5 and len_condition(src_words,
                                                                                                    dst_words):
            w.write(src.strip() + " ||| " + dst.strip() + "\n")
            written += 1

        print(written, "/", i, end="\r")

print("\nDone!")
