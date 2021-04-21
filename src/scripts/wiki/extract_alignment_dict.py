import os
import sys
from collections import Counter, defaultdict

input_path = os.path.abspath(sys.argv[1])
alignment_path = os.path.abspath(sys.argv[2])
output_path = os.path.abspath(sys.argv[3])

src_word_counts = Counter()
src2dst_count = defaultdict(Counter)

with open(input_path, "r") as reader, open(alignment_path, "r") as areader, open(output_path, "w") as w:
    used = 0
    for i, (line, aline) in enumerate(zip(reader, areader)):
        spl = line.strip().split(" ||| ")
        src_words = spl[0].strip().split(" ")
        dst_words = spl[1].strip().split(" ")

        src_word_counts += Counter(src_words)

        for a in aline.strip().split(" "):
            try:
                s, t = int(a.split("-")[0]), int(a.split("-")[1])
                src2dst_count[src_words[s]][dst_words[t]] += 1
            except:
                pass

        if (i + 1) % 1000 == 0:
            print(i + 1, end="\r")

    print("\nCalculating!")
    for src_word in src2dst_count.keys():
        dst_counter: Counter = src2dst_count[src_word]
        sorted = dst_counter.most_common()
        sc = src_word_counts[src_word]

        output = [src_word]
        for s in sorted:
            output.append(s[0])
            output.append(str(s[1] / sc))
        w.write("\t".join(output))
        w.write("\n")

print("\nDone!")
