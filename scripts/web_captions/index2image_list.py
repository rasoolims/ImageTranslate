import os
import sys

index_file = os.path.abspath(sys.argv[1])
dir = os.path.abspath(sys.argv[2])
output_file = os.path.abspath(sys.argv[3])

extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
with open(index_file, "r") as r, open(output_file, "w") as w:
    for line in r:
        spl = line.strip().split("\t")
        file_name = os.path.join(dir, spl[0])
        for extension in extensions:
            if os.path.exists(file_name + extension):
                file_name = file_name + extension
                break
        w.write(file_name + "\t" + spl[-1] + "\n")
