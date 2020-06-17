import os
import sys

path = os.path.abspath(sys.argv[1])

docs, sens = 0, 0
types = set()
with open(path, "r") as reader:
    for line in reader:
        docs += 1
        sens += len(line.strip().split("</s>"))
        types |= set(line.strip().split(" "))
        print(docs, end="\r")

print(docs, "docs,", sens, "sens,", len(types), "types")
