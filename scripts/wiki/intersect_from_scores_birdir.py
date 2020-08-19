import os
import sys

input_path = os.path.abspath(sys.argv[1])
output_path = os.path.abspath(sys.argv[2])
forward_dict = dict()

print("forward")
with open(input_path, "r") as reader:
    for i, line in enumerate(reader):
        forward = line.strip().split("\t")
        forward_dict[forward[0]] = (forward[1], float(forward[2]))
        if (i + 1) % 1000 == 0:
            print(i + 1, end="\r")
print("\nbackward")
found = 0
with open(output_path, "w") as w:
    for src in forward_dict.keys():
        dst, p1 = forward_dict[src]
        if dst in forward_dict and forward_dict[dst] == src:
            _, p2 = forward_dict[dst]
            w.write(src + " ||| " + dst + "\t" + str(p1) + "\t" + str(p2) + "\t" + str(p1 + p2) + "\n")
            found += 1
        if (i + 1) % 1000 == 0:
            print(found, "/", i + 1, end="\r")

print("\ndone!")
