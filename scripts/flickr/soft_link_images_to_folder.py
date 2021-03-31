import os
import sys

output_folder = os.path.abspath(sys.argv[2])
with open(os.path.abspath(sys.argv[1]), "r") as reader:
    for line in reader:
        spl = line.strip().split("\t")
        if len(spl) < 2: continue

        command = "ln -s {0} {1}".format(spl[0], output_folder)
        os.system(command)
