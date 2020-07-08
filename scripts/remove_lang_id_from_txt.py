import os
import sys

input_file = os.path.abspath(sys.argv[1])
output_file = os.path.abspath(sys.argv[2])

with open(input_file, "r") as reader, open(output_file, "w") as writer:
    for line in reader:
        sen = " ".join(line.strip().split(" ")[1:-1])
        writer.write(sen + "\n")
