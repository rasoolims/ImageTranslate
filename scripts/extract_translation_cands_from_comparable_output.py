import os
import sys

"""
From output of comprable data best translation candidates, only saves those that are the same from both language
directions.
"""

input_path = os.path.abspath(sys.argv[1])
output_path = os.path.abspath(sys.argv[2])

translation_dict = {}

print("Reading translation candidates!")
with open(input_path, "r") as reader:
    for line in reader:
        spl = line.strip().split("\t")
        if len(spl) != 3: continue

        translation_dict[spl[0]] = (spl[1], spl[2])

print("Getting shared dictionary")
shared_dict = {}
for s1 in translation_dict.keys():
    s2, p1 = translation_dict[s1]
    if s2 not in translation_dict: continue
    if s2 in shared_dict: continue

    s3, p2 = translation_dict[s2]

    if s3.lower().strip() == s1.lower().strip():
        shared_dict[s1] = (s2, p1, p2)

print("Writing shared dictionary")
with open(output_path, "w") as writer:
    for s1 in shared_dict.keys():
        s2, p1, p2 = shared_dict[s1]
        writer.write("\t".join([s1, s2, p1, p2]))
        writer.write("\n")
print("Done!")
