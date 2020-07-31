import os
import sys

input_path = os.path.abspath(sys.argv[1])
output_path1 = os.path.abspath(sys.argv[2])
output_path2 = os.path.abspath(sys.argv[3])

with open(input_path, "r") as reader, open(output_path1, "w") as w1, open(output_path2, "w") as w2:
    for line in reader:
        spl = line.strip().split(" ||| ")
        if len(spl) != 2: continue
        w1.write(spl[0] + "\n")
        w2.write(" ".join(spl[1:]).strip() + "\n")
