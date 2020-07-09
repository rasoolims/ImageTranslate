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

print(len(src), len(gold), len(translation))
open(src_path, "w").write("\n".join(src) + "\n")
open(gold_path, "w").write("\n".join(gold) + "\n")
open(translation_path, "w").write("\n".join(translation) + "\n")
