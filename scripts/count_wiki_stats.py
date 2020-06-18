import os
import sys

path = os.path.abspath(sys.argv[1])

docs, sens, toks = 0, 0, 0
types = set()
with open(path, "r") as reader:
    for line in reader:
        docs += 1
        sens += len(line.strip().split("</s>"))
        words = line.strip().split(" ")
        toks += len(words) - 2
        types |= set(words)
        print(docs, end="\r")

print(docs, "docs,", sens, "sens,", len(types), "types,", toks, "tokens")
