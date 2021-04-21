import os
import sys

input_path = os.path.abspath(sys.argv[1])
lang_id = "<" + sys.argv[2] + ">"
eos = "</s>"
output_path = os.path.abspath(sys.argv[3])

with open(output_path, "w") as writer, open(input_path, "r") as reader:
    for line in reader:
        line = lang_id + " " + line.strip() + " " + eos + "\n"
        writer.write(line)
