import os
import sys
from collections import defaultdict

import sacrebleu

ref = defaultdict(list)
max_ref = 0
with open(os.path.abspath(sys.argv[1]), "r") as ref_reader:
    for line in ref_reader:
        spl = line.strip().split("\t")
        if len(spl) < 2: continue
        path = spl[0].strip()
        ref[path].append(spl[1].strip())
        max_ref = max(max_ref, len(ref[path]))

references = [[] for _ in range(max_ref)]
outputs = []
with open(os.path.abspath(sys.argv[2]), "r") as output_reader:
    for line in output_reader:
        spl = line.strip().split("\t")
        if len(spl) < 2: continue
        path = spl[0].strip()
        output = spl[1].strip()
        ref_values = ref[path]
        for i in range(max_ref):
            if i < len(ref_values):
                references[i].append(ref_values[i])
            else:
                references[i].append(None)
        outputs.append(output)

bleu = sacrebleu.corpus_bleu(outputs, references, lowercase=True, tokenize="intl")
print(bleu)
print(bleu.score)
