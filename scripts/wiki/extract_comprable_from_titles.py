import os
import sys

print("Reading titles")
title_dict = {}
with open(os.path.abspath(sys.argv[1]), "r") as title_reader:
    for line in title_reader:
        a, e = line.strip().split("\t")
        if "(" in a:
            a = a[:a.find("(")]
        if "(" is e:
            e = e[:e.find("(")]

        title_dict[e] = a

print("Reading source docs")

src_docs = {}
with open(os.path.abspath(sys.argv[2]), "r") as src_reader:
    for i, line in enumerate(src_reader):
        sentences = line.strip().split("</s>")
        title = sentences[0][sentences[0].find(">") + 1:].strip()

        src_docs[title] = sentences[1:]
        print(i, end="\r")

print("\nReading target docs")
found = 0
with open(os.path.abspath(sys.argv[3]), "r") as dst_reader, open(os.path.abspath(sys.argv[4]), "w") as src_writer, open(
        os.path.abspath(sys.argv[5]), "w") as dst_writer:
    for line in dst_reader:
        sentences = line.strip().split("</s>")
        title = sentences[0][sentences[0].find(">") + 1:].strip()

        if title in title_dict:
            src_title = title_dict[title]
            if src_title in src_docs:
                src_sentences = src_docs[src_title]

                src_writer.write("\t".join(src_sentences) + "\n")
                dst_writer.write("\t".join(sentences[1:]) + "\n")
                found += 1
        print(found, "/", i, end="\r")
print("\nDone!")
