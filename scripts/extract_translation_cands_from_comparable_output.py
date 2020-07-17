import os
import sys

"""
From output of comprable data best translation candidates, only saves those that are the same from both language
directions.
"""

input_path = os.path.abspath(sys.argv[1])
output_path = os.path.abspath(sys.argv[2])
limit = -13  # lowest possible value for sum of log probs
if len(sys.argv) > 3:
    limit = float(sys.argv[3])

translation_dict = {}

print("Reading translation candidates!")
with open(input_path, "r") as reader:
    for line in reader:
        spl = line.strip().split("\t")
        if len(spl) != 3: continue

        translation_dict[spl[0]] = (spl[1], spl[2])

print("Getting shared dictionary")
shared_dict = {}
first_lang = None
for s1 in translation_dict.keys():
    lang = s1.strip().split(" ")[0]
    if first_lang is None:
        first_lang = lang
    s2, p1 = translation_dict[s1]
    if s2 not in translation_dict: continue
    if s2 in shared_dict: continue

    s3, p2 = translation_dict[s2]

    if s3.lower().strip() == s1.lower().strip():
        p12 = float(p1) + float(p2)
        if p12 >= limit and lang == first_lang:
            shared_dict[s1] = (s2, p1, p2, str(p12))

print("Writing shared dictionary")
with open(output_path, "w") as writer:
    for s1 in shared_dict.keys():
        s2, p1, p2, p12 = shared_dict[s1]
        s1_lang =
        writer.write("\t".join([s1, s2, p1, p2, p12]))
        writer.write("\n")
print("Done!")
