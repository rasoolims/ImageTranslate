import os
import sys

input_path = os.path.abspath(sys.argv[1])
output_path = os.path.abspath(sys.argv[2])
src_path = output_path + ".src"
gold_path = output_path + ".gold"
translation_path = output_path + ".trans"

content = open(input_path, "r").read().strip().split("\n")

src, gold, translation = [], [], []
for i, c in enumerate(content):
    if (i + 1) % 6 == 1:
        src.append(c)
    elif (i + 1) % 6 == 2:
        translation.append(c)
    elif (i + 1) % 6 == 3:
        gold.append(c)
    else:
        pass

open(src_path, "w").write("\n".join(src))
open(gold_path, "w").write("\n".join(gold))
open(translation_path, "w").write("\n".join(translation))
