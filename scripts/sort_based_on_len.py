import os
import sys
from collections import defaultdict

len_dict = defaultdict(set)
with open(os.path.abspath(sys.argv[1]), "r") as r:
    for line in r:
        line = line.strip()
        ln = len(line.split(" "))
        len_dict[ln].add(line)

with open(os.path.abspath(sys.argv[2]), "w") as w:
    for ln in sorted(len_dict.keys()):
        w.write("\n".join(len_dict[ln]))
        w.write("\n")
