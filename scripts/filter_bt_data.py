import os
import re
import sys

has_number = lambda i: bool(re.search(r'\d', i))
len_condition = lambda words1, words2: True if abs(len(words1) - len(words2)) <= 5 else False

src_file = os.path.abspath(sys.argv[1])
dst_file = os.path.abspath(sys.argv[2])
punc_letters = sys.argv[3]
output_file = os.path.abspath(sys.argv[4])
wrote = 0
with open(src_file, "r") as r1, open(dst_file, "r") as r2, open(output_file, "w") as w:
    for i, (s, t) in enumerate(zip(r1, r2)):
        s = s.strip()
        t = t.strip()

        sw = s.split(" ")
        tw = t.split(" ")

        ns = has_number(s)
        nt = has_number(t)

        num_consistent = (ns and nt) or not (ns or nt)

        if num_consistent and len_condition(sw, tw):
            if s.endswith(".") and not t.endswith("."):
                t += punc_letters[0]
            elif s.endswith("!") and not t.endswith("!"):
                t += punc_letters[1]
            elif s.endswith("?") and not t.endswith("?"):
                t += punc_letters[2]
            w.write(s + " ||| " + t + "\n")
            wrote += 1

        if i % 1000 == 0:
            print(wrote, "/", i, end="\r")
print("\nDone!")
