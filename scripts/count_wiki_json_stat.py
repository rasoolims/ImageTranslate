import json
import os
import sys

path = os.path.abspath(sys.argv[1])

docs, sens = 0, 0
types, images = set(), set()
with open(path, "rb") as fp:
    contents = json.load(fp)
    for content in contents:
        docs += 1
        sens += len(content["content"].strip().split("</s>"))
        types |= set(content["content"].strip().split(" "))
        images |= set(map(lambda img: img["img_path"], content["images"]))
        print(docs, end="\r")

print(docs, "docs,", sens, "sens,", len(types), "types", len(images), "images")
