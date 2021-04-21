from extract_caption_candidates_from_html_files import *


def download_dir(titles, image_dict, fp, num_written, input_folder, fasttext_model, lang):
    print(input_folder)
    dirs = os.listdir(input_folder)
    for f, file in enumerate(dirs):
        file_path = os.path.join(input_folder, file)
        if file_path.endswith(".html") or file_path.endswith(".htm"):
            num_written = download(titles, image_dict, file_path, fp, num_written, fasttext_model, lang)
            if (f + 1) % 100 == 0:
                print(input_folder, f + 1, "/", len(dirs), "-> wrote", num_written)  # , end="\r")
        elif os.path.isdir(file_path):
            num_written = download_dir(titles, image_dict, fp, num_written, file_path, fasttext_model, lang)
    return num_written


input_folder = os.path.abspath(sys.argv[1])
fasttext_model = fasttext.load_model(os.path.abspath(sys.argv[2]))
lang = "__label__" + sys.argv[3]
titles = get_titles(os.path.abspath(sys.argv[4]))
output_file = os.path.abspath(sys.argv[5])

num_written = 0

image_dict = {}
with open(output_file, "w") as fp:
    num_written = download_dir(titles, image_dict, fp, num_written, input_folder, fasttext_model, lang)
print("\nWrote", num_written)
