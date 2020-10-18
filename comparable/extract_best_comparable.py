from optparse import OptionParser


def get_option_parser():
    parser = OptionParser()
    parser.add_option("--src", dest="src_file", metavar="FILE", default=None)
    parser.add_option("--dst", dest="dst_file", metavar="FILE", default=None)
    parser.add_option("--src-tok", dest="src_tok_file", metavar="FILE", default=None)
    parser.add_option("--dst-tok", dest="dst_tok_file", metavar="FILE", default=None)
    parser.add_option("--scores", dest="score_file", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_file", metavar="FILE", default=None)
    parser.add_option("--min", dest="min_sim", type="float", default=0.1)
    return parser


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


if __name__ == "__main__":
    parser = get_option_parser()
    (options, args) = parser.parse_args()

    highest_s2d = dict()
    highest_d2s = dict()

    print("Reading scores")
    with open(options.src_file, "r") as sr, open(options.dst_file, "r") as dr, open(options.src_tok_file, "r") as stokr, open(options.dst_tok_file, "r") as dtokr, open(options.score_file, "r") as scf:
        for i, (src_line, dst_line, score_line, stok_line, dtok_line) in enumerate(zip(sr, dr, scf, stokr, dtokr)):
            src_line = src_line.strip()
            dst_line = dst_line.strip()
            if not number_match(stok_line.strip(), dtok_line.strip()):
                continue

            score = float(score_line.strip())

            if src_line not in highest_s2d:
                highest_s2d[src_line] = (dst_line, score)
            elif score > highest_s2d[src_line][1]:
                highest_s2d[src_line] = (dst_line, score)

            if dst_line not in highest_d2s:
                highest_d2s[dst_line] = (src_line, score)
            elif score > highest_d2s[dst_line][1]:
                highest_d2s[dst_line] = (src_line, score)

            if i % 10000 == 0:
                print(i, end="\r")

    print("\nWriting shared highest scores")
    found = 0
    shared_dict = dict()
    with open(options.output_file, "w") as w:
        for i, src_line in enumerate(highest_s2d.keys()):
            dst_line, score = highest_s2d[src_line]

            if i % 10000 == 0:
                print(found, "/", i, end="\r")

            if highest_d2s[dst_line][0] == src_line and score >= options.min_sim:
                shared_dict[src_line + " ||| " + dst_line] = score
                found += 1
        sorted = sorted(shared_dict.items(), key=lambda x: x[1], reverse=True)
        for (sen, score) in sorted:
            w.write(sen + "\t" + str(score) + "\n")

    print("\nDone!")
