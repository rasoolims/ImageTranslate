import os
import sys

path = lambda i: os.path.abspath(sys.argv[i])

# define punctuation
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~؛،؟!'''

with open(path(1), "r") as r, open(path(2), "w") as w:
    for line in r:
        no_punct = []
        for char in line.strip():
            if char not in punctuations:
                no_punct.append(char)
        w.write("".join(no_punct))
        w.write("\n")
