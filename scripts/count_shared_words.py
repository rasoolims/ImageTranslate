import os
import sys

s1, s2 = set(), set()
with open(os.path.abspath(sys.argv[1]), "r") as r:
    for line in r:
        s1 += set(line.strip().split(" "))

with open(os.path.abspath(sys.argv[2]), "r") as r:
    for line in r:
        s2 += set(line.strip().split(" "))

print(len(s1), len(s2), len(s1 & s2))
