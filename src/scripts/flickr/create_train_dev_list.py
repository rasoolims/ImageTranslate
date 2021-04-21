import os
import sys

caption_txt_f = os.path.abspath(sys.argv[1])
split_folder = os.path.abspath(sys.argv[2])
image_folder = os.path.abspath(sys.argv[3])
output_file = os.path.abspath(sys.argv[4])

with open(os.path.join(split_folder, "Flickr_8k.trainImages.txt")) as r:
    train_paths = set(map(lambda x: x.strip(), r))
with open(os.path.join(split_folder, "Flickr_8k.devImages.txt")) as r:
    dev_paths = set(map(lambda x: x.strip(), r))
with open(os.path.join(split_folder, "Flickr_8k.testImages.txt")) as r:
    test_paths = set(map(lambda x: x.strip(), r))

with open(caption_txt_f, "r") as r, open(output_file + ".train.en", "w") as train_en, \
        open(output_file + ".dev.en", "w") as dev_en, open(output_file + ".test.en", "w") as test_en:
    for line in r:
        spl = line.strip().split(",")
        if spl[0] != "image":
            path, caption = os.path.join(image_folder, spl[0]), " ".join(spl[1:])
            if spl[0] in train_paths:
                train_en.write(path + "\t" + caption + "\n")
            elif spl[0] in dev_paths:
                dev_en.write(path + "\t" + caption + "\n")
            elif spl[0] in test_paths:
                test_en.write(path + "\t" + caption + "\n")
with open(os.path.join(split_folder, "Flickr8k.arabic.full.txt"), "r") as r, \
        open(output_file + ".train.ar", "w") as train_ar, open(output_file + ".dev.ar", "w") as dev_ar, \
        open(output_file + ".test.ar", "w") as test_ar:
    for line in r:
        spl = line.strip().split("\t")
        if spl[0] != "image":
            path, caption = os.path.join(image_folder, spl[0][:-2]), " ".join(spl[1:])
            if spl[0][:-2] in train_paths:
                train_ar.write(path + "\t" + caption + "\n")
            elif spl[0][:-2] in dev_paths:
                dev_ar.write(path + "\t" + caption + "\n")
            elif spl[0][:-2] in test_paths:
                test_ar.write(path + "\t" + caption + "\n")
