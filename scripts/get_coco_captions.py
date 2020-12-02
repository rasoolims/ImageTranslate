import json
import os
import sys
from collections import defaultdict

input_path = os.path.abspath(sys.argv[1])
image_folder = os.path.abspath(sys.argv[2])
output_path = os.path.abspath(sys.argv[3])

with open(input_path, "r") as r:
    obj = json.load(r)

annotations = obj["annotations"]

caption_dict = defaultdict(list)
with open(output_path, "w") as w:
    for annotation in annotations:
        caption = annotation["caption"].strip()
        image_path = str(annotation["image_id"])
        added_zeros = "".join((12 - len(image_path)) * ["0"])
        image_path = "".join([added_zeros, image_path, ".jpg"])
        caption_dict[image_path].append(caption)
    print(len(caption_dict))
    max_len = 0
    min_len = 1000
    for image_path in caption_dict.keys():
        output = [os.path.join(image_folder,image_path)] + caption_dict[image_path]
        w.write("\t".join(output))
        w.write("\n")
        max_len = max(len(caption_dict[image_path]), max_len)
        min_len = min(len(caption_dict[image_path]), min_len)
print(len(annotations))
print(max_len)
print(min_len)
