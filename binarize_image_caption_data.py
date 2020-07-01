import marshal
import os
from optparse import OptionParser

from PIL import Image
from torchvision import transforms

from textprocessor import TextProcessor


def write(text_processor: TextProcessor, output_file: str, input_file: str, root_img_dir, skip_check: bool = False,
          max_len: int = 256):
    transform = transforms.Compose([  # [1]
        transforms.Resize(256),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])

    skipped_long_sens = 0
    image_path_dict, unique_images = dict(), dict()
    with open(os.path.join(input_file), "rb") as fp:
        captions = marshal.load(fp)

        tok_captions = {}
        image_ids = {}
        for c in captions:
            try:
                tok_sen = text_processor.tokenize_one_sentence(c[1])
                if len(tok_sen) > max_len:
                    skipped_long_sens += 1
                    continue

                path = c[0]
                if not skip_check and path not in image_path_dict:
                    with Image.open(os.path.join(root_img_dir, path)) as im:
                        # make sure not to deal with rgba or grayscale images.
                        _ = transform(im.convert("RGB"))
                        im.close()
                    image_id = len(unique_images)
                    unique_images[image_id] = path
                    image_path_dict[path] = image_id
                if skip_check and path not in image_path_dict:
                    image_id = len(unique_images)
                    unique_images[image_id] = path
                    image_path_dict[path] = image_id
                elif path in image_path_dict:
                    image_id = image_path_dict[path]
                    unique_images[image_id] = path

                caption_id = len(tok_captions)
                tok_captions[caption_id] = tok_sen
                image_ids[caption_id] = image_id
            except:
                pass

        print("Skipped long sentences:", skipped_long_sens)
        tok_captions_sorted = sorted(tok_captions.items(), key=lambda item: len(item[1]))
        caption_sorted = list(map(lambda e: (image_ids[e[0]], e[1]), tok_captions_sorted))
        print("Longest sentence", len(tok_captions_sorted[-1][1]))
        with open(output_file, "wb") as wfp:
            marshal.dump((unique_images, caption_sorted), wfp)
        print("Dumped", len(caption_sorted), "captions from", len(unique_images), "unique images")


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
