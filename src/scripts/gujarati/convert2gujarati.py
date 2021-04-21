import os
import sys

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

input_text = os.path.abspath(sys.argv[1])
output_text = os.path.abspath(sys.argv[2])
with open(input_text, "r") as r, open(output_text, "w") as w:
    for i, line in enumerate(r):
        line = line.strip()
        w.write(transliterate(line, sanscript.DEVANAGARI, sanscript.GUJARATI))
        w.write("\n")
        if i % 10000 == 0:
            print(i, end="\r")
print("\nDone!")
