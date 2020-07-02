import datetime
import marshal
import os
from optparse import OptionParser

from PIL import Image
from torchvision import transforms

from textprocessor import TextProcessor

transform = transforms.Compose([  # [1]
    transforms.Resize(256),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
    transforms.Normalize(  # [5]
        mean=[0.485, 0.456, 0.406],  # [6]
        std=[0.229, 0.224, 0.225]  # [7]
    )])


def is_valid_img(root_img_dir, path):
    try:
        with Image.open(os.path.join(root_img_dir, path)) as im:
            # make sure not to deal with rgba or grayscale images.
            _ = transform(im.convert("RGB"))
            im.close()
        return True
    except:
        return False


def write(text_processor: TextProcessor, output_file: str, input_file: str, root_img_dir, skip_check: bool = False,
          max_len: int = 256):
    skipped_long_sens = 0
    image_path_dict, unique_images = dict(), dict()
    with open(os.path.join(input_file), "rb") as fp:
        captions = marshal.load(fp)

        print(datetime.datetime.now(), "Tokenizing sentences")
        tok_sens = list(map(lambda c: text_processor.tokenize_one_sentence(c[1]), captions))
        print(datetime.datetime.now(), "Checking images")
        img_pths = list(map(lambda c: c[0], captions))
        uniq_img_paths = set(img_pths)
        valid_images = set(
            filter(lambda x: x != None, map(lambda i: i if is_valid_img(root_img_dir, i) else None, uniq_img_paths)))
        unique_images = {k: i for k, i in enumerate(valid_images)}
        path_ids = {i: k for k, i in enumerate(valid_images)}

        print(datetime.datetime.now(), "Getting file captions")
        captid = lambda i, s: (s, path_ids[img_pths[i]]) if len(s) <= max_len and img_pths[i] in valid_images else None
        tok_captions = list(filter(lambda x: x != None, map(lambda i, s: captid(i, s), range(len(captions)), tok_sens)))
        print(datetime.datetime.now(), "Dumping...")
        with open(output_file, "wb") as wfp:
            marshal.dump((unique_images, tok_captions), wfp)
        print(datetime.datetime.now(), "Dumped", len(tok_captions), "captions from", len(unique_images),
              "unique images")


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--file", dest="file", help="Which files to use", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_file", help="Output pickle file.", metavar="FILE", default=None)
    parser.add_option("--image", dest="image_dir", help="Root image directory", metavar="FILE", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--max-len", dest="max_len", help="Maximum tokenized caption length", type="int", default=256)
    parser.add_option("--skip-check", action="store_true", dest="skip_check",
                      help="Skipping checking if image file exists", default=False)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    tokenizer = TextProcessor(options.tokenizer_path)

    print("Writing batches")
    write(text_processor=tokenizer,
          output_file=options.output_file,
          input_file=options.file,
          root_img_dir=options.image_dir,
          skip_check=options.skip_check,
          max_len=options.max_len)
    print("Finished")
