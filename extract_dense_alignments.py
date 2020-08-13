import os
import sys

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

        density = min(len(src_words) / len(alignments), len(dst_words) / len(alignments))

        if density >= min_density and len(src_words) >= 5 and len(dst_words) >= 5:
            w.write(src.strip() + "|||" + dst.strip() + "\n")

        print(written, "/", i, end="\r")

print("\nDone!")
