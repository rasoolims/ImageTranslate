import os
import re
import sys

import fasttext
from bs4 import BeautifulSoup

en_chars = set(list("abcdefghijklmnopqrstuvwxyz"))
banned_words = {"blog", "thumb", "logo", "small", "banner", "slide"}
banned_puncts = {"–", ":", "_", "-", "\"", "»", "«", "…", "ـ", "?", "؟", "!", "!", "+", ")", "(", "[", "]"}


def has_english(inputString):
    return any(map(lambda x: x in en_chars, list(inputString.lower())))


def good_size(src):
    # Find if size is hidden in text
    result = re.search(r'[0-9]+x[0-9]+', src.lower())
    if result is None:
        return True
    sizes = result.group(0).split("x")
    x, y = int(sizes[0]), int(sizes[1])
    return x >= 256 and y >= 256


def contains_number(inputString):
    return any(char.isdigit() for char in inputString)


def contains_english(inputString):
    return any(char.isalphanum() for char in inputString)


def is_title(inputString, title_set):
    for title in title_set:
        if title in inputString:
            return True
    return False


def download(titles, image_dict, file_path, fp, num_written, fasttext_model):
    try:
        content = open(file_path, "r").read()
        soup = BeautifulSoup(content, 'html.parser')
        images = soup.find_all("img")
        if len(images) == 0: return num_written
        image_info = list(filter(lambda x: x is not None, map(lambda im: img_info(im, lang, titles, fasttext_model), images)))
        if len(image_info) == 0: return num_written

        alt_text = []
        for src, alt in image_info:
            if src not in image_dict:
                if "|" in alt:
                    alt = alt[alt.rfind("|") + 1:].strip()
                    if len(alt.split(" ")) < 5:
                        continue
                if is_title(alt, titles):
                    continue
                alt_text.append(alt + "\t" + src)
                image_dict[src] = alt

        if len(alt_text) == 0: return num_written
        fp.write("\n".join(alt_text))
        fp.write("\n")
        num_written += len(alt_text)
    except:
        pass
    return num_written


lang_condition = lambda alt, lang, fasttext_model: fasttext_model.predict(alt)[0][0] == lang
alt_condition = lambda alt, lang, titles, fasttext_model: len(alt.strip().split(" ")) > 5 and not contains_number(
    alt) and not has_english(
    alt) and "." not in alt[:-1] and "." not in alt[:-1] and all(
    map(lambda x: x not in alt, banned_puncts)) and lang_condition(alt, lang, fasttext_model)
good_format = lambda src: src.endswith(".jpg") or src.endswith(".png") or src.endswith(".jpeg")
src_condition = lambda src: good_format(src.strip().lower()) and good_size(src) and all(
    map(lambda x: x not in src.lower(), banned_words))
img_con = lambda im: im["alt"] is not None and len(im["alt"].strip()) > 1 and src_condition(im["src"])
img_info = lambda im, lang, titles, fasttext_model: (im["src"].strip(), im["alt"]) if img_con(im) and alt_condition(im["alt"], lang,
                                                                                                    titles, fasttext_model) else None
get_titles = lambda path: set(filter(lambda x: len(x.split(" ")) >= 2,
                                     map(lambda x: x[
                                                   :x.find("(") + 1].strip().lower() if "(" in x else x.strip().lower(),
                                         open(path, "r").read().strip().split("\n"))))



if __name__ == "__main__":
    input_folder = os.path.abspath(sys.argv[1])
    fasttext_model = fasttext.load_model(os.path.abspath(sys.argv[2]))
    lang = "__label__" + sys.argv[3]
    titles = get_titles(os.path.abspath(sys.argv[4]))
    output_file = os.path.abspath(sys.argv[5])

    num_written = 0
    image_dict = {}

    with open(output_file, "w") as fp:
        dirs = os.listdir(input_folder)
        for f, file in enumerate(dirs):
            file_path = os.path.join(input_folder, file)
            num_written = download(titles, image_dict, file_path, fp, num_written, fasttext_model)
            if (f + 1) % 100 == 0:
                print(f + 1, "/", len(dirs), "-> wrote", num_written , end="\r")
    print("\nWrote", num_written)
