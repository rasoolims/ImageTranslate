import os
import sys

ap = lambda x: os.path.abspath(x)

with open(ap(sys.argv[1]), "r") as r, open(ap(sys.argv[2]), "w") as w1, open(ap(sys.argv[3]), "w") as w2:
    for line in r:
        spl = [s.strip() for s in line.strip().split("\t")]
        for i in range(1, len(spl)):
            if len(spl[0]) > 1 and len(spl[1]) > 1:
                w1.write(spl[0] + "\n")
                w2.write(spl[i] + "\n")
