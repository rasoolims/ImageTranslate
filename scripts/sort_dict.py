import os
import sys

threshold = float(sys.argv[3])

pair_dict = {}
with open(os.path.abspath(sys.argv[1]), "r") as r:
    for line in r:
        spl = line.strip().split("\t")

        if len(spl) == 3 and float(spl[2]) >= threshold and spl[1].lower() != spl[2].lower():
            if "." in spl[0] or "." in spl[1] or spl[0][0].isupper() or spl[1][0].isupper():
                continue
            pair_dict[spl[0] + "\t" + spl[1]] = float(spl[2])

pair_dict = sorted(pair_dict.items(), key=lambda x: x[1], reverse=True)
covered = set()
with open(os.path.abspath(sys.argv[2]), "w") as w:
    for x, y in pair_dict:
        s, t = x.split("\t")
        if s not in covered:
            covered.add(s)
            w.write(x + "\t" + str(y) + "\n")
