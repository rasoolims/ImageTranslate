import os
import sys

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


input_text = os.path.abspath(sys.argv[1])
output_text = os.path.abspath(sys.argv[2])
with open(input_text, "r") as r, open(output_text, "w") as w:
    for i, line in enumerate(r):
        line = line.strip()
        converted = []
        w.write("".join(
            map(lambda c: c if isEnglish(c) else transliterate(c, sanscript.ITRANS, sanscript.GUJARATI), list(line))))
        w.write("\n")
        if i % 10000 == 0:
            print(i, end="\r")
print("\nDone!")
