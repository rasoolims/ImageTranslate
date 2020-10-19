from optparse import OptionParser


def get_option_parser():
    parser = OptionParser()
    parser.add_option("--src", dest="src_file", metavar="FILE", default=None)
    parser.add_option("--dst", dest="dst_file", metavar="FILE", default=None)
    parser.add_option("--scores", dest="score_file", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_file", metavar="FILE", default=None)
    parser.add_option("--min", dest="min_sim", type="float", default=0.1)
    return parser


if __name__ == "__main__":
    parser = get_option_parser()
    (options, args) = parser.parse_args()

    highest_s2d = dict()
    highest_d2s = dict()

    print("Reading scores")
    with open(options.src_file, "r") as sr, open(options.dst_file, "r") as dr, open(options.score_file, "r") as scf:
        for i, (src_line, dst_line, score_line) in enumerate(zip(sr, dr, scf)):
            src_line = src_line.strip()
            dst_line = dst_line.strip()

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
