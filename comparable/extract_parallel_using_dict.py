import os
import sys
from collections import defaultdict


def sim(s1, s2, src2dst_dict):
    ws1 = s1.strip().split(" ")
    ws2 = s2.strip().split(" ")
    found = 0
    for w1 in ws1:
        for w2 in ws2:
            if w2 in src2dst_dict[w1] or w1 == w2:
                found += 1
    return found / min(len(ws1), len(ws2))


print("Reading dictionary")
src2dst_dict = defaultdict(set)
with open(os.path.abspath(sys.argv[1]), "r") as dict_reader:
    for line in dict_reader:
        spl = line.strip().split("\t")
        src2dst_dict[spl[0]].add(spl[1])
        src2dst_dict[spl[0]].add(spl[1].lower())

print("Reading corpus")
src2dst_sen_max_sim = dict()
dst2src_sen_max_sim = dict()
with open(os.path.abspath(sys.argv[2]), "r") as corpus_reader:
    for i, line in enumerate(corpus_reader):
        spl = line.strip().split(" ||| ")
        sen_sim = sim(spl[0].lower(), spl[1].lower(), src2dst_dict)
        if sen_sim > 0.3:
            if spl[0] not in src2dst_sen_max_sim:
                src2dst_sen_max_sim[spl[0]] = (spl[1], sen_sim)
            elif src2dst_sen_max_sim[spl[0]][1] < sen_sim:
                src2dst_sen_max_sim[spl[0]] = (spl[1], sen_sim)

            if spl[1] not in dst2src_sen_max_sim:
                dst2src_sen_max_sim[spl[1]] = (spl[0], sen_sim)
            elif dst2src_sen_max_sim[spl[1]][1] < sen_sim:
                dst2src_sen_max_sim[spl[1]] = (spl[0], sen_sim)
        if i % 10000 == 0:
            print(i, end="\r")

print("\nGetting best results")
wrote = 0
with open(os.path.abspath(sys.argv[3]), "w") as writer:
    for i, src_sen in enumerate(src2dst_sen_max_sim.keys()):
        dst_sen, sen_sim = src2dst_sen_max_sim[src_sen]
        if dst2src_sen_max_sim[dst_sen][0] == src_sen:
            writer.write(src_sen + " ||| " + dst_sen + "\t" + str(sen_sim) + "\n")
            wrote += 1
        print(i, wrote, end="\r")
print("\nDone!")
