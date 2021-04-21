import os
import sys

min_len = int(sys.argv[3])
max_len = int(sys.argv[4])

wrote = 0
with open(os.path.abspath(sys.argv[1]), "r") as r1, open(os.path.abspath(sys.argv[2]), "r") as r2, open(
        os.path.abspath(sys.argv[5]), "w") as w1, open(os.path.abspath(sys.argv[6]), "w") as w2:
    for i, (s, t) in enumerate(zip(r1, r2)):
        s = s.strip()
        t = t.strip()
        spl1 = s.split(" ")
        spl2 = t.split(" ")
        if min_len <= len(spl1) <= max_len and min_len <= len(spl2) <= max_len:
            wrote += 1
            w1.write(s + "\n")
            w2.write(t + "\n")

        if i % 10000 == 0:
            print(wrote, "/", i, end="\r")

print("\nDone", wrote)
