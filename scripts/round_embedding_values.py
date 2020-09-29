import os
import sys

with open(os.path.abspath(sys.argv[1]), "r") as r, open(os.path.abspath(sys.argv[2]), "w") as w:
    for line in r:
        spl = line.strip().split(" ")
        for i in range(1, len(spl)):
            spl[i] = str(round(float(spl[i]), 4))
        w.write(" ".join(spl) + "\n")
