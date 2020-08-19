import marshal
import os
import re
import sys
from collections import defaultdict

has_number = lambda i: bool(re.search(r'\d', i))
len_condition = lambda l1, l2: True if .8 <= l1 / l2 <= 1.2 or abs(l1 - l2) <= 5 and l1 >= 5 and l2 >= 5 else False

print("Reading titles")
title_dict = {}
with open(os.path.abspath(sys.argv[1]), "r") as title_reader:
    for line in title_reader:
        spl = line.strip().split("\t")
        if len(spl) != 2:
            continue
        a, e = spl
        if "(" in a:
            a = a[:a.find("(")]
        if "(" is e:
            e = e[:e.find("(")]

        title_dict[e] = a

print("Number of titles", len(title_dict))
print("Reading source docs")

sen_ids = dict()
sen_lens = dict()

src_docs = {}
with open(os.path.abspath(sys.argv[2]), "r") as src_reader:
    for i, line in enumerate(src_reader):
        sentences = line.strip().split("</s>")
        title = sentences[0][sentences[0].find(">") + 1:].strip()
        sens = []
        if len(sentences) < 4:
            continue
        for sen in sentences[1:]:
            if sen not in sen_ids:
                sen = sen.replace("()", "").replace("  ", " ").strip()
                sen_lens[len(sen_ids)] = len(sen.split(" "))
                sen_ids[sen] = len(sen_ids)
            sens.append(sen_ids[sen])

        src_docs[title] = sens
        if i % 1000 == 0: print(i, end="\r")

print("\nReading target docs")
src2dst_dict = defaultdict(set)
dst2src_dict = defaultdict(set)

found = 0
with open(os.path.abspath(sys.argv[3]), "r") as dst_reader:
    for i, line in enumerate(dst_reader):
        sentences = line.strip().split("</s>")
        title = sentences[0][sentences[0].find(">") + 1:].strip()
        sens = []
        if len(sentences) < 4:
            continue
        for sen in sentences[1:]:
            if sen not in sen_ids:
                sen = sen.replace("()", "").replace("  ", " ").strip()
                sen_lens[len(sen_ids)] = len(sen.split(" "))
                sen_ids[sen] = len(sen_ids)
            sens.append(sen_ids[sen])

        if title in title_dict:
            src_title = title_dict[title]
            if src_title in src_docs:
                src_sentences = src_docs[src_title]
                for tgt_sen in sens:
                    for src_sen in src_sentences:
                        if len_condition(sen_lens[src_sen], sen_lens[tgt_sen]):
                            src2dst_dict[src_sen].add(tgt_sen)
                            dst2src_dict[tgt_sen].add(src_sen)

                found += 1
        if i % 1000 == 0:
            print(found, "/", i, end="\r")

to_del = set()
for sen in src2dst_dict.keys():
    if len(src2dst_dict[sen]) <= 2:
        to_del.add(sen)
print("\nDeleting", len(to_del))
for sen in to_del:
    del src2dst_dict[sen]
    del sen_ids[sen]

to_del = set()
for sen in dst2src_dict.keys():
    if len(dst2src_dict[sen]) <= 2:
        to_del.add(sen)
        try:
            del sen_ids[sen]
        except:
            pass

print("Deleting", len(to_del))
for sen in to_del:
    del dst2src_dict[sen]

with open(sys.argv[4], "wb") as writer:
    print("\nWriting", len(sen_ids), len(src2dst_dict), len(dst2src_dict))
    marshal.dump((sen_ids, dict(src2dst_dict), dict(dst2src_dict)), writer)

print("Done!")
