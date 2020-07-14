import json
import os
import sys

path = os.path.abspath(sys.argv[1])
min_len = int(sys.argv[2])
max_len = int(sys.argv[3])
output_path = os.path.abspath(sys.argv[4])

len_condition = lambda s: len(s.strip()) > 0 and max_len >= len(s.strip().split(" ")) >= min_len
with open(path, "rb") as fp, open(output_path, "w") as writer:
    doc_dicts = json.load(fp)

    for i, v in enumerate(doc_dicts):
        if len(v["images"]) > 0:
            content_spl = v["content"].strip().split(" ")
            lang_id, content = content_spl[0] + " ", " ".join(content_spl[1:])
            sens = list(filter(lambda x: x != None,
                               map(lambda s: " ".join([lang_id, s.strip(), "</s>"]) if len_condition(s) else None,
                                   content.split("</s>"))))
            writer.write("\n".join(sens))
            writer.write("\n")
        print(i, "/", len(doc_dicts), end="\r")

print("Done!")
