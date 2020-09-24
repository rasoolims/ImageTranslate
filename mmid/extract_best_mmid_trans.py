import os

from optparse import OptionParser


def get_option_parser():
    parser = OptionParser()
    parser.add_option("--input", dest="input_file", metavar="FILE", default=None)
    parser.add_option("--root", dest="root_path", metavar="FILE", default="")
    parser.add_option("--output", dest="output_file", metavar="FILE", default=None)
    return parser


if __name__ == "__main__":
    parser = get_option_parser()
    (options, args) = parser.parse_args()

    src2dst = dict()
    dst2src = dict()

    print("Reading sims")
    with open(options.input_file, "r") as r:
        for i, line in enumerate(r):
            spl = line.strip().split("\t")
            src_path, dst_path, sim = spl[1], spl[2], float(spl[2])
            if not src_path.startswith("/"):
                src_path = os.path.join(options.root_path, src_path)
            if not dst_path.startswith("/"):
                dst_path = os.path.join(options.root_path, dst_path)

            src_word = open(os.path.join(src_path, "word.txt"), "r").read().strip()
            dst_word = open(os.path.join(dst_path, "word.txt"), "r").read().strip()

            if src_word not in src2dst:
                src2dst[src_word] = (dst_word, sim)
            elif sim > src2dst[src_word][1]:
                src2dst[src_word] = (dst_word, sim)

            if dst_word not in dst2src:
                dst2src[dst_word] = (src_word, sim)
            elif sim > dst2src[src_word][1]:
                dst2src[dst_word] = (src_word, sim)

            if i % 100000 == 0:
                print((i / 1000000), "M", len(src2dst), len(dst2src), end="\r")

    print("Writing sims")
    with open(options.output_file, "w") as w:
        for i, src_word in enumerate(src2dst):
            dst_word = src2dst[src_word][0]
            if dst2src[dst_word][0] == src_word:
                w.write(src_word + "\t" + dst_word + "\t" + str(src2dst[src_word][1]) + "\n")
            if i % 100000 == 0:
                print(i, end="\r")
    print("Done!")
