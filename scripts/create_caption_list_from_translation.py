import os
import sys

translations = dict()
with open(os.path.abspath(sys.argv[1]), "r") as r:
    for line in r:
        spl = line.strip().split("|||")
        translations[spl[0].strip()] = spl[1].strip()
print("Extracted", len(translations), "unique translations!")

with open(os.path.abspath(sys.argv[2]), "r") as r, open(os.path.abspath(sys.argv[3]), "w") as w:
    for line in r:
        spl = line.strip().split("\t")
        if spl[1] in translations:
            w.write(spl[0] + "\t" + translations[spl[1]] + "\n")
