import os
import sys

input_path = os.path.abspath(sys.argv[1])
lowest = float(sys.argv[2])
lowest_sum = float(sys.argv[3])
output_path = os.path.abspath(sys.argv[4])
forward_dict = dict()

print("forward")
with open(input_path, "r") as reader:
    for i, line in enumerate(reader):
        try:
            forward = line.strip().split("\t")
            p = float(forward[2])
            if 8 <= len(forward[0].split(" ")) <= 50 and 8 <= len(forward[1].split(" ")) <= 50 and p >= lowest:
                forward_dict[forward[0]] = (forward[1], p)
            if (i + 1) % 1000 == 0:
                print(i + 1, end="\r")
        except:
            pass
print("\nbackward")
found = 0
with open(output_path, "w") as w:
    for src in forward_dict.keys():
        dst, p1 = forward_dict[src]

        if dst in forward_dict and forward_dict[dst][0] == src:
            _, p2 = forward_dict[dst]

            l1 = src.split(" ")[0]
            l2 = dst.split(" ")[0]

            psum = p1 + p2
            if psum < lowest_sum:
                continue

            if l1 > l2:
                w.write(dst + " ||| " + src + "\t" + str(p1) + "\t" + str(p2) + "\t" + str(psum) + "\n")
            else:
                w.write(src + " ||| " + dst + "\t" + str(p1) + "\t" + str(p2) + "\t" + str(psum) + "\n")
            found += 1
        if (i + 1) % 1000 == 0:
            print(found, "/", i + 1, end="\r")

print("\ndone!")
