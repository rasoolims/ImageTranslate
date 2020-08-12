import os
import sys

pair_dict = {}
with open(os.path.abspath(sys.argv[1]), "r") as r:
    for line in r:
        spl = line.strip().split("\t")
        if len(spl) == 3:
            pair_dict[spl[0] + "\t" + spl[1]] = float(spl[2])

pair_dict = sorted(pair_dict.items(), key=lambda x: x[1], reverse=True)
with open(os.path.abspath(sys.argv[2]), "w") as w:
    for x, y in pair_dict:
        w.write(x + "\t" + str(y) + "\n")
