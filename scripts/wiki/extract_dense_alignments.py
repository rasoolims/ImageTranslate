import os
import sys

input_path = os.path.abspath(sys.argv[1])
alignment_path = os.path.abspath(sys.argv[2])
proportion = float(sys.argv[3])
output_path = os.path.abspath(sys.argv[4])

with open(input_path, "r") as reader, open(alignment_path, "r") as areader, open(output_path, "w") as w:
    used = 0
    for i, (line, aline) in enumerate(zip(reader, areader)):
        spl = line.strip().split(" ||| ")
        src_len = len(spl[0].strip().split(" "))
        dst_len = len(spl[1].strip().split(" "))
        sen_len = min(src_len, dst_len)
        alen = len(aline.strip().split(" "))
        if alen / sen_len >= proportion:
            w.write(line.strip() + "\n")
            used += 1
        if i % 1000 == 0:
            print(used, "/", i + 1, end="\r")
print("\nDone!")
