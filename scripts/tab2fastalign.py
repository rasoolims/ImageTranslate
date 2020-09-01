import os
import sys

input_path = os.path.abspath(sys.argv[1])
output_path = os.path.abspath(sys.argv[2])

with open(input_path, "r") as reader, open(output_path, "w") as w:
    for line in reader:
        spl = line.strip().split("\t")
        if len(spl) < 2: continue
        w.write(spl[0] + " ||| " + spl[1] + "\n")
