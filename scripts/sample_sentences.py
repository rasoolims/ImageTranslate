import os
import random
import sys

input_file = os.path.abspath(sys.argv[1])
line_count = int(sys.argv[2])
output_file = os.path.abspath(sys.argv[3])

sentences = []
with open(input_file, "r") as reader:
    for line in reader:
        sentences.append(line.strip())

random.shuffle(sentences)

with open(output_file, "r") as writer:
    writer.write("\n".join(sentences[:line_count]))
