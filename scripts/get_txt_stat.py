import os
import sys

path = os.path.abspath(sys.argv[1])

docs, sens, toks = 0, 0, 0
types = set()
with open(path, "r") as reader:
    for line in reader:
        line = line.strip()
        sens += 1
        words = line.split(" ")
        toks += len(words) - 2 if words[0].startswith("<") else len(words)
        types |= set(words)
        print(sens, end="\r")

print(sens, "sens,", len(types), "types,", toks, "tokens")
