import os
import re
import sys

has_number = lambda i: bool(re.search(r'\d', i))
len_condition = lambda words1, words2: True if .7 <= len(words1) / len(words2) <= 1.3 or abs(
    len(words1) - len(words2)) <= 5 and len(words1) >= 5 and len(words2) >= 5 else False

print("\nReading docs")
found = 0
with open(os.path.abspath(sys.argv[1]), "r") as src_reader, open(os.path.abspath(sys.argv[2]), "r") as dst_reader, open(
        os.path.abspath(sys.argv[3]), "w") as writer:
    for i, (src, dst) in enumerate(zip(src_reader, dst_reader)):
        src = src.strip().replace(" </s> ", " ")
        dst = dst.strip().replace(" </s> ", " ")
        if src.endswith("</s>"):
            src = " ".join(src.split(" ")[1:-1]).strip()
        if dst.endswith("</s>"):
            dst = " ".join(dst.split(" ")[1:-1]).strip()

        n1 = has_number(src)
        n2 = has_number(dst)
        src_words = src.split(" ")
        dst_words = dst.split(" ")

        if len_condition(src_words, dst_words):
            if (n1 and n2) or (not n1 and not n2):
                writer.write(src + " ||| " + dst + "\n")
                found += 1

        print(found, "/", i, end="\r")
print("\nDone!")
