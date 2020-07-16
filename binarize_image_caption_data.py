import json
import marshal
import os
from itertools import chain
from optparse import OptionParser

from PIL import Image
from torchvision import transforms

from textprocessor import TextProcessor

"""
Extracts all images with all sentences (longer than 5 words).
"""
sen_chooser = lambda sens, img: list(map(lambda s: (img, s), sens))
img_sen_collect = lambda image, sens: [(image["img_path"], image["caption"])] + sen_chooser(sens, image["img_path"])
ref_sen_chooser = lambda i, s, sens, r, img: (img["img_path"], sens[i]) if s > r else None


def extract_captions(v, ref_images=None):
    if ref_images is not None:
        shared_images = list(filter(lambda x: x is True, map(lambda img: img["img_path"] in ref_images, v["images"])))
        if len(shared_images) == 0:
            return []
    return list(map(lambda i: (i["img_path"], i["caption"]), v["images"]))


def extract_shared_sentences(v, ref_images=None):
    if ref_images is not None:
        shared_images = list(filter(lambda x: x is True, map(lambda img: img["img_path"] in ref_images, v["images"])))
        if len(shared_images) == 0:
            return []
    content_spl = v["content"].strip().split(" ")
    lang_id, content = content_spl[0] + " ", " ".join(content_spl[1:])
    sens = list(map(lambda s: lang_id + s.strip() + " </s>", content.split("</s>")))
    sen_words = list(map(lambda s: set(s.split()[1:-1]), sens))
    return list(chain(*map(lambda image: extract_captions4imgs(image, sen_words, sens), v["images"])))


def extract_captions4imgs(image, sen_words, sens):
    caption = image["caption"]
    caption_words = set(caption.strip().split(" ")[1:-1])
    shared_word_counts = list(map(lambda s: len(s & caption_words), sen_words))
    max_word_count = max(shared_word_counts)
    least_req_count = max(2, max_word_count - 2)
    captions = [(image["img_path"], caption)] + list(
        filter(lambda x: x != None,
               map(lambda i, s: ref_sen_chooser(i, s, sens, least_req_count, image), range(len(sens)),
                   shared_word_counts)))
    return captions


def extract_sentences(v, ref_images=None):
    if ref_images is not None:
        shared_images = list(filter(lambda x: x is True, map(lambda img: img["img_path"] in ref_images, v["images"])))
        if len(shared_images) == 0:
            return []
    content_spl = v["content"].strip().split(" ")
    lang_id, content = content_spl[0] + " ", " ".join(content_spl[1:])
    sens = list(filter(lambda x: x != None,
                       map(lambda s: lang_id + s.strip() + " </s>" if len(s.strip().split(" ")) >= 5 else None,
                           content.split("</s>"))))
    result = list(chain(*map(lambda img: img_sen_collect(img, sens), v["images"])))
    return result


def write(text_processor: TextProcessor, output_file: str, input_file: str, root_img_dir, skip_check: bool = False,
          max_len: int = 256, ref_file=None, choose_relevant=True, only_captions=False):
    ref_images = None
    if ref_file is not None:
        with open(ref_file, "rb") as fp:
            doc_dicts = json.load(fp)
            ref_images = set(chain(*map(lambda v: list(map(lambda im: im["img_path"], v["images"])), doc_dicts)))

    with open(input_file, "rb") as fp:
        doc_dicts = json.load(fp)
        num_captions = sum(list(map(lambda v: len(v["images"]), doc_dicts)))
        if only_captions:
            captions = list(chain(*map(lambda v: extract_captions(v, ref_images), doc_dicts)))
        elif choose_relevant:
            captions = list(chain(*map(lambda v: extract_shared_sentences(v, ref_images), doc_dicts)))
        else:
            captions = list(chain(*map(lambda v: extract_sentences(v, ref_images), doc_dicts)))
        print(num_captions, len(captions))

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

    tok_captions = {}
    image_ids = {}
    for ci, c in enumerate(captions):
        if ci % 1000 == 0:
            print(ci, "/", len(captions), "->", len(tok_captions), len(unique_images), end="\r")
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

    print("Skipped long sentences:", skipped_long_sens, "from", len(captions))
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
    parser.add_option("--ref", dest="ref", help="Ref files to use for overlap", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_file", help="Output pickle file.", metavar="FILE", default=None)
    parser.add_option("--image", dest="image_dir", help="Root image directory", metavar="FILE", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--max-len", dest="max_len", help="Maximum tokenized caption length", type="int", default=256)
    parser.add_option("--skip-check", action="store_true", dest="skip_check",
                      help="Skipping checking if image file exists", default=False)
    parser.add_option("--all", action="store_true", dest="use_all",
                      help="Choose all sentences instead of only subset of captions", default=False)
    parser.add_option("--only", action="store_true", dest="only", help="Choose only captions", default=False)
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
          max_len=options.max_len,
          ref_file=options.ref,
          only_captions=options.only,
          choose_relevant=not options.use_all)
    print("Finished")
