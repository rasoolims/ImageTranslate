import json
import os
import sys

path1 = os.path.abspath(sys.argv[1])
path2 = os.path.abspath(sys.argv[2])

docs1, sens1 = 0, 0
types1, images1 = set(), set()
captions1 = 0
with open(path1, "rb") as fp:
    contents = json.load(fp)
    for content in contents:
        docs1 += 1
        sens1 += len(content["content"].strip().split("</s>"))
        types1 |= set(content["content"].strip().split(" "))
        image_list = list(map(lambda img: img["img_path"], content["images"]))
        images1 |= set(image_list)
        captions1 += len(image_list)
        print(docs1, end="\r")

print(docs1, "docs,", sens1, "sens,", len(types1), "types,", len(images1), "images,", captions1, "captions")

docs2, sens2 = 0, 0
captions2 = 0
types2, images2 = set(), set()
with open(path2, "rb") as fp:
    contents = json.load(fp)
    for content in contents:
        docs2 += 1
        sens2 += len(content["content"].strip().split("</s>"))
        types2 |= set(content["content"].strip().split(" "))
        image_list = list(map(lambda img: img["img_path"], content["images"]))
        images2 |= set(image_list)
        captions2 += len(image_list)
        print(docs2, end="\r")

print(docs2, "docs,", sens2, "sens,", len(types2), "types,", len(images2), "images,", captions2, "captions")

shared_images = images1 & images2
print("Shared images:", len(shared_images))
