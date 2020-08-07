from extract_caption_candidates_from_html_files import *

fasttext_model = fasttext.load_model(os.path.abspath(sys.argv[2]))
lang = "__label__" + sys.argv[3]
titles = get_titles(os.path.abspath(sys.argv[4]))

with open(os.path.abspath(sys.argv[1]), "r") as r, open(os.path.abspath(sys.argv[5]), "w") as w:
    for ic, caption_line in enumerate(r):
        path, caption = caption_line.strip().split("\t")
        caption = caption.replace(" </s> ", " ")
        sen = " ".join(caption.strip().split(" ")[1:-1])

        if alt_condition(sen, lang, titles,fasttext_model):
            w.write(path+"\t"+caption+"\n")
        if ic%10000==0:
            print(ic)