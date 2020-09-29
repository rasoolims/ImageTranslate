import os
import sys

src_words = set()
dst_words = set()

dict_entries = list()
with open(os.path.abspath(sys.argv[1]), "r") as r, open(os.path.abspath(sys.argv[2]), "r2") as r2, open(
        os.path.abspath(sys.argv[3]), "w") as w:
    for line in r:
        spl = line.strip().split("\t")
        src_words.add(spl[0])
        dst_words.add(spl[1])
        w.write(line.strip() + "\n")
    for line in r:
        spl = line.strip().split("\t")
        if spl[0] not in src_words and spl[1] not in dst_words:
            w.write(line.strip() + "\n")
