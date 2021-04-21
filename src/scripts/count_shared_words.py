import os
import sys

s1, s2 = set(), set()
c1, c2 = set(), set()
with open(os.path.abspath(sys.argv[1]), "r") as r:
    for line in r:
        s1 |= set(line.strip().split(" "))
        c1 |= set(list(line.strip()))

with open(os.path.abspath(sys.argv[2]), "r") as r:
    for line in r:
        s2 |= set(line.strip().split(" "))
        c2 |= set(list(line.strip()))

print(len(s1), len(s2), len(s1 & s2))
print(len(c1), len(c2), len(c1 & c2))
print(c1)
print(c2)
print(c1 & c2)
