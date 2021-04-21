import os
import sys

replacements = {"۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4", "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
                "٫": ".", "૦": "0", "०": "0", "૧": "1", "१": "1", "૨": "2", "२": "2", "૩": "3", "३": "3", "૪": "4",
                "४": "4", "૫": "5", "५": "5", "૬": "6", "६": "6", "૭": "7", "७": "7", "૮": "8", "८": "8", "૯": "9",
                "९": "9"}

tok_replacements = {}


def digit_replace(tok):
    if tok in tok_replacements:
        return tok_replacements[tok]
    new_tok = "".join(map(lambda char: replacements[char] if char in replacements else char, list(tok)))
    tok_replacements[tok] = new_tok
    return new_tok


is_digit = lambda x: x.replace('.', '', 1).isdigit()


def number_match(src_txt, dst_txt):
    src_words = src_txt.split(" ")
    dst_words = dst_txt.split(" ")
    digit_src = set(filter(lambda x: is_digit(x), map(lambda x: digit_replace(x), src_words)))
    digit_dst = set(filter(lambda x: is_digit(x), map(lambda x: digit_replace(x), dst_words)))
    return digit_dst == digit_src


removed = 0
with open(os.path.abspath(sys.argv[1]), "r") as r, open(os.path.abspath(sys.argv[2]), "r") as r1, open(
        os.path.abspath(sys.argv[3]), "r") as r2, open(os.path.abspath(sys.argv[4]), "w") as w:
    for line, src, dst in zip(r, r1, r2):
        line = line.strip()
        if number_match(src.strip(), dst.strip()):
            w.write(line + "\n")
        else:
            removed += 1

print(removed)
