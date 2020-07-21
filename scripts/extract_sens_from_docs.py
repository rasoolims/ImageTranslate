import os
import sys


def extract_doctences(line, min_len, max_len):
    line = line.strip()
    if len(line) == 0:
        return []

    docs = line.split("</s>")
    doc_split = docs[0].strip().split(" ")
    docs[0] = " ".join(doc_split[1:])
    lang = doc_split[0]
    len_condition = lambda s: len(s.strip()) > 0 and max_len >= len(s.strip().split(" ")) >= min_len
    return list(filter(lambda x: x is not None,
                       map(lambda s: " ".join([lang, s.strip(), "</s>"]) if len_condition(s) else None, docs)))


path = os.path.abspath(sys.argv[1])
min_len = int(sys.argv[2])
max_len = int(sys.argv[3])
output_path = os.path.abspath(sys.argv[4])

with open(path, "r") as reader, open(output_path, "w") as writer:
    for i, line in enumerate(reader):
        sens = extract_doctences(line, min_len, max_len)
        if len(sens) > 0:
            writer.write("\n".join(sens))
            writer.write("\n")
        if i % 1000 == 0:
            print(i, end="\r")
print("\nDone!")
