import os
import sys

threshold = float(sys.argv[3])
min_len = int(sys.argv[4])

pair_dict = {}
with open(os.path.abspath(sys.argv[1]), "r") as r:
    for line in r:
        spl = line.strip().split("\t")

        if len(spl) == 3 and float(spl[2]) >= threshold and spl[0].lower().strip() != spl[1].lower().strip():
            if "." in spl[0] or "." in spl[1]:  # or spl[0][0].isupper() or spl[1][0].isupper():
                continue
            if len(spl[0].split(" ")) < min_len or len(spl[1].split(" ")) < min_len:
                continue
            pair_dict[spl[0].strip() + "\t" + spl[1].strip()] = float(spl[2])

pair_dict = sorted(pair_dict.items(), key=lambda x: x[1], reverse=True)
covered = set()
with open(os.path.abspath(sys.argv[2]), "w") as w:
    for x, y in pair_dict:
        s, t = x.split("\t")
        if s not in covered:
            covered.add(s)
            w.write(x + "\t" + str(y) + "\n")
