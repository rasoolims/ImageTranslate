import os
import sys

output_folder = os.path.abspath(sys.argv[2])
files = set()
with open(os.path.abspath(sys.argv[1]), "r") as reader:
    for line in reader:
        spl = line.strip().split("\t")
        if len(spl) < 2: continue

        if spl[0] not in files:
            command = "ln -s {0} {1}".format(spl[0], output_folder)
            os.system(command)
            files.add(spl[0])
