import os
import sys

with open(os.path.abspath(sys.argv[1]), "r") as r, open(os.path.abspath(sys.argv[2]), "w") as w:
    for i, line in enumerate(r):
        spl = line.strip().split(" ")
        spl[1:] = list(map(lambda x: str(round(float(x), 4)), spl[1:]))
        w.write(" ".join(spl) + "\n")
        print(i, end="\r")
print("\nDone!")
