import os
import sys

us = set()
with open(os.path.abspath(sys.argv[1]), "r") as r:
    for i, l in enumerate(r):
        us.add(l)
        if i % 10000 == 0:
            print(i, end="\r")

print("\n", len(us))
