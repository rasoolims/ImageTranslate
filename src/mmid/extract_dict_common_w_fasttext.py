import os
import sys

ar_embedding_path = os.path.abspath(sys.argv[1])
en_embedding_path = os.path.abspath(sys.argv[2])
dict_path = os.path.abspath(sys.argv[3])
output_path = os.path.abspath(sys.argv[4])

print("Reading source")
ar_words = set()
with open(ar_embedding_path, "r") as ar:
    for line in ar:
        ar_words.add(line.strip().split(" ")[0])

print("Reading English")
en_words = set()
with open(en_embedding_path, "r") as en:
    for line in en:
        en_words.add(line.strip().split(" ")[0])

print("Reading/Writng dictionary")
with open(dict_path, "r") as din, open(output_path, "w") as dout:
    for line in din:
        words = line.strip().split("\t")
        w = words[0] if words[0] in ar_words else words[0].lower()
        if w in ar_words:
            for word in words[1:]:
                if word in en_words:
                    dout.write(w + " ||| " + word + "\n")
                elif word.lower() in en_words:
                    dout.write(w + " ||| " + word.lower() + "\n")
                else:
                    print(word)
