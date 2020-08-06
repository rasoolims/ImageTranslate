import os
import sys

input_list = open(os.path.abspath(sys.argv[1]), "r").read().strip().split("\n")
output_folder = os.path.abspath(sys.argv[2])
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
log_file = os.path.join(output_folder, "log.txt")

with open(os.path.join(output_folder, "list.txt"), "w") as writer:
    for i, url in enumerate(input_list):
        writer.write(str(i) + "\t" + url + "\n")

        command = ["wget -k --tries=1 --timeout=5", url, "-O", os.path.join(output_folder, str(i)), "-o", log_file]
        if (i + 1) % 100 != 0:
            command.append("&")
        else:
            print((i + 1), "/", len(input_list), end="\r")

        os.system(" ".join(command))
print("\nDone!")
