import os
import sys

min_len = int(sys.argv[2])
max_len = int(sys.argv[3])

wrote = 0
with open(os.path.abspath(sys.argv[1]), "r") as r, open(os.path.abspath(sys.argv[4]), "w") as w:
    for i, s in enumerate(r):
        s = s.strip()
        spl = s.split(" ")
        if min_len <= len(spl) <= max_len:
            wrote += 1
            w.write(s + "\n")

        if i % 10000 == 0:
            print(wrote, "/", i, end="\r")

print("\nDone", wrote)
