import os
import re
import sys

guj_digits = {"૦", "૧", "૨", "૩", "૪", "૫", "૬", "૭", "૮", "૯", "०", "१", "२", "३", "४", "५", "६", "७", "८", "९"}

has_number = lambda i: bool(re.search(r'\d', i)) or any(map(lambda x: x in guj_digits, i))
len_condition = lambda words1, words2: True if (.7 <= len(words1) / len(words2) <= 1.3 or abs(
    len(words1) - len(words2)) <= 5) and len(words1) >= 5 and len(words2) >= 5 else False

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
j = 0
k = 1
with open(os.path.abspath(sys.argv[3]), "r") as dst_reader, open(os.path.abspath(sys.argv[4]), "w") as first_sen_writer:
    for i, line in enumerate(dst_reader):
        sentences = line.strip().split("</s>")
        title = sentences[0][sentences[0].find(">") + 1:].strip()

        if title in title_dict:
            src_title = title_dict[title]
            if src_title in src_docs:
                src_sentences = src_docs[src_title]
                if sentences[k].lower().startswith("early life"):
                    continue  # Common phrase in Wiki
                if "list of" in sentences[k].lower():
                    continue  # Common phrase in Wiki
                n1 = has_number(sentences[k])
                n2 = has_number(src_sentences[j])
                sen_words2 = sentences[k].strip().split(" ")
                sen_words1 = sentences[j].strip().split(" ")
                if not len_condition(sen_words1, sen_words2):
                    continue
                if (n1 and n2) or (not n1 and not n2):
                    if src_sentences[j].lower() != sentences[k].lower():
                        first_sen_writer.write(src_sentences[j] + " ||| " + sentences[k] + "\n")
                        found += 1

        print(found, "/", i, end="\r")
print("\nDone!")
