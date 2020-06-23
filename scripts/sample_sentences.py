import os
import random
import sys

input_file = os.path.abspath(sys.argv[1])
line_count = int(sys.argv[2])
output_file = os.path.abspath(sys.argv[3])

sentences = []
with open(input_file, "r") as reader:
    for i, line in enumerate(reader):
        sentences.append(line.strip())
        print(i, end="\r")

random.shuffle(sentences)

with open(output_file, "w") as writer:
    writer.write("\n".join(sentences[:line_count]))
print("\nDone!")
