import os
import sys

print("\nReading docs")
found = 0
with open(os.path.abspath(sys.argv[1]), "r") as reader, open(os.path.abspath(sys.argv[2]), "w") as writer:
    for i, line in enumerate(reader):
        src, dst = line.strip().split("\t")
        if "(" not in src and "(" not in dst:
            writer.write(src + " ||| " + dst + "\n")
            found += 1

        print(found, "/", i, end="\r")
print("\nDone!")
