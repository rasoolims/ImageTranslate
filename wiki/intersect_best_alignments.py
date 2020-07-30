import os
import sys

input_path = os.path.abspath(sys.argv[1])
rev_path = os.path.abspath(sys.argv[2])
output_path = os.path.abspath(sys.argv[3])
forward_dict = dict()

print("forward")
with open(input_path, "r") as reader:
    for i, line in enumerate(reader):
        forward = line.strip().split("\t")
        spl = forward[0].strip().split(" ||| ")
        forward_dict[spl[0]] = (spl[1], float(forward[1]))
        if (i + 1) % 1000 == 0:
            print(i + 1, end="\r")
print("\nbackward")
found = 0
with open(rev_path, "r") as rev_reader, open(output_path, "w") as w:
    for i, line in enumerate(rev_reader):
        backward = line.strip().split("\t")
        spl = backward[0].strip().split(" ||| ")
        reverse, fprob = forward_dict[spl[1]]
        if reverse.strip() == spl[0]:
            prob = str(float(backward[1]) * fprob)
            w.write(spl[1] + " ||| " + spl[0] + "\t" + str(prob) + "\n")
            found += 1
        if (i + 1) % 1000 == 0:
            print(found, "/", i + 1, end="\r")

print("\ndone!")
