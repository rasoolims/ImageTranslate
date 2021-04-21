import os
import sys

input_path = os.path.abspath(sys.argv[1])
alignment_path = os.path.abspath(sys.argv[2])
output_path = os.path.abspath(sys.argv[3])
a_output_path = os.path.abspath(sys.argv[4])

with open(input_path, "r") as reader, open(alignment_path, "r") as areader, open(output_path, "w") as w, open(
        a_output_path, "w") as aw:
    for i, (line, aline) in enumerate(zip(reader, areader)):
        spl = line.strip().split(" ||| ")
        outputline = spl[1] + " ||| " + spl[0]
        w.write(outputline + "\n")

        align_output = []
        for a in aline.strip().split(" "):
            try:
                aspl = a.strip().split("-")
                align_output.append(aspl[1] + "-" + aspl[0])
            except:

                pass
        aw.write(" ".join(align_output) + "\n")
        if (i + 1) % 1000 == 0:
            print(i + 1, end="\r")
