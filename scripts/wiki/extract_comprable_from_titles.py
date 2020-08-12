import os
import sys

len_condition = lambda words1, words2: True if .7 <= len(words1) / len(words2) <= 1.3 or abs(
    len(words1) - len(words2)) <= 5 and len(words1) >= 5 and len(words2) >= 5 else False

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
        os.path.abspath(sys.argv[5]), "w") as dst_writer, open(os.path.abspath(sys.argv[6]), "w") as first_sen_writer:
    for i, line in enumerate(dst_reader):
        sentences = line.strip().split("</s>")
        title = sentences[0][sentences[0].find(">") + 1:].strip()

        if title in title_dict:
            src_title = title_dict[title]
            if src_title in src_docs:
                src_sentences = src_docs[src_title]

                src_writer.write("\t".join(src_sentences) + "\n")
                dst_writer.write("\t".join(sentences[1:]) + "\n")

                sen_words1 = src_sentences[0].strip().split(" ")
                sen_words2 = sentences[1].strip().split(" ")
                if len_condition(sen_words1, sen_words2):
                    src_sentences[0] = src_sentences[0].replace("()", "").replace("  ", " ").strip()
                    sentences[1] = sentences[1].replace("()", "").replace("  ", " ").strip()
                    first_sen_writer.write(src_sentences[0] + "\t" + sentences[1] + "\n")
                found += 1
        print(found, "/", i, end="\r")
print("\nDone!")
