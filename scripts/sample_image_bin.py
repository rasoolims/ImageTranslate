import marshal
import os
import random
import sys

"""
If the number of images is imbalanced between two languages, use the smallest language set as the reference size.
"""

input_file = os.path.abspath(sys.argv[1])
output_file = os.path.abspath(sys.argv[2])

with open(input_file, "rb") as fp:
    lang_specific_images, unique_images, unique_docs = marshal.load(fp)

min_len = min([len(v) if l != "shared" else float("inf") for l, v in lang_specific_images.items()])
languages = set(lang_specific_images.keys()) - {"shared"}

print([(l, len(v)) for l, v in lang_specific_images.items()])
print(min_len)

for lang in languages:
    v = lang_specific_images[lang]
    if len(v) > min_len:
        keys = list(lang_specific_images[lang].keys())
        random.shuffle(keys)
        keys_to_use = keys[:min_len]
        new_v = {}
        for k in keys_to_use:
            new_v[k] = lang_specific_images[lang][k]
        lang_specific_images[lang] = new_v

print("Getting new unique images")
remaining_images = [set(v.keys()) for _, v in lang_specific_images.items()]
images_to_use = set.union(*remaining_images)
new_unique_images = dict()
for im in images_to_use:
    new_unique_images[im] = unique_images[im]

print("Getting new unique documents")
new_docs = dict()
for l, entry in lang_specific_images.items():
    for v in entry.values():
        for t in v:
            new_docs[t[2]] = unique_docs[t[2]]

print("Dumping")
with open(output_file, "wb") as fp:
    marshal.dump((lang_specific_images, new_unique_images, new_docs), fp)

print("Done!")
