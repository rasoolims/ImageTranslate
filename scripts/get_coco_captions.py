import json
import os
import sys

input_path = os.path.abspath(sys.argv[1])
output_path = os.path.abspath(sys.argv[2])

with open(input_path, "r") as r:
    obj = json.load(r)

annotations = obj["annotations"]

with open(output_path, "w") as w:
    for annotation in annotations:
        caption = " ".join(["<en>", annotation["caption"], "</s>"])
        image_path = str(annotation["image_id"])
        added_zeros = "".join((12-len(image_path))*["0"])
        image_path = "".join([added_zeros, image_path, ".jpg"])
        w.write(image_path + "\t" + caption + "\n")
print(len(annotations))
