import marshal
import os
import re
import sys
from collections import defaultdict

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~؛،؟!'''


def remove_punc(sentence):
    for char in punctuations:
        sentence = sentence.replace(char, " ")
    sentence = " ".join(sentence.split())
    return sentence


has_number = lambda i: bool(re.search(r'\d', i))
len_condition = lambda l1, l2: True if abs(l1 - l2) <= 5 and 50 >= l1 >= 8 and 50 >= l2 >= 8 else False

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
            ln = len(sen.split(" "))
            if 8 <= ln <= 50:
                sens.append((sen, ln))

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
        if title in title_dict:
            sens = []
            if len(sentences) < 4:
                continue

            src_title = title_dict[title]
            if src_title in src_docs:
                src_sentences = src_docs[src_title]

                for tgt_sen in sentences[1:]:
                    tgt_ln = len(tgt_sen.split(" "))
                    if not (8 <= tgt_ln <= 50):
                        continue
                    for (src_sen, ln) in src_sentences:
                        if len_condition(ln, tgt_ln):
                            if src_sen not in sen_ids:
                                src_sen = remove_punc(src_sen)
                                sen_ids[src_sen] = len(sen_ids)
                            if tgt_sen not in sen_ids:
                                tgt_sen = remove_punc(tgt_sen)
                                sen_ids[tgt_sen] = len(sen_ids)

                            src2dst_dict[sen_ids[src_sen]].add(sen_ids[tgt_sen])
                            dst2src_dict[sen_ids[tgt_sen]].add(sen_ids[src_sen])

                found += 1
        if i % 1000 == 0:
            print(found, "/", i, end="\r")

to_del = set()
for sen in src2dst_dict.keys():
    if len(src2dst_dict[sen]) == 1:
        to_del.add(sen)
print("\nDeleting", len(to_del))
for sen in to_del:
    del src2dst_dict[sen]

to_del = set()
for sen in dst2src_dict.keys():
    if len(dst2src_dict[sen]) == 1:
        to_del.add(sen)

print("Deleting", len(to_del))
for sen in to_del:
    del dst2src_dict[sen]

uniq_sens = None

with open(sys.argv[4] + ".sens", "wb") as writer1, open(sys.argv[4] + ".src", "wb") as writer2, open(
        sys.argv[4] + ".dst", "wb") as writer3:
    print("\nWriting", len(sen_ids), len(src2dst_dict), len(dst2src_dict))
    marshal.dump(list(sen_ids.keys()), writer1)
    marshal.dump(dict(src2dst_dict), writer2)
    marshal.dump(dict(dst2src_dict), writer3)

print("Done!")
