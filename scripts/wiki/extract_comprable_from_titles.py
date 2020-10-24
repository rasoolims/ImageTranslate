import os
import re
import sys

has_number = lambda i: bool(re.search(r'\d', i))
len_condition = lambda words1, words2: True if ((.3 <= len(words1) / len(words2) <= 3 or abs(
    len(words1) - len(words2))) <= 5) and len(words1) >= 5 and len(words2) >= 5 else False

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

                for j in range(len(src_sentences)):
                    # region1 = j / len(src_sentences)
                    src_sentences[j] = src_sentences[j].replace("()", "").replace("  ", " ").strip()
                    sen_words1 = src_sentences[j].strip().split(" ")
                    n2 = has_number(src_sentences[j])
                    for k in range(1, len(sentences)):
                        # region2 = (k - 1) / (len(sentences) - 1)
                        # if abs(region1 - region2) > 0.3:
                        #     continue
                        sentences[k] = sentences[k].replace("()", "").replace("  ", " ").strip()
                        sen_words2 = sentences[k].strip().split(" ")
                        if len_condition(sen_words1, sen_words2):
                            if sentences[k].lower().startswith("early life"):
                                continue  # Common phrase in Wiki
                            if "list of" in sentences[k].lower():
                                continue  # Common phrase in Wiki
                            n1 = has_number(sentences[k])
                            if (n1 and n2) or (not n1 and not n2):
                                if src_sentences[j].lower() == sentences[k].lower():
                                    continue
                                if j == 0 and k == 1:
                                    first_sen_writer.write(src_sentences[j] + "\t" + sentences[k] + "\n")
                                src_writer.write(src_sentences[j] + "\n")
                                dst_writer.write(sentences[k] + "\n")
                                found += 1

        print(found, "/", i, end="\r")
print("\nDone!")
