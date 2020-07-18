import os
import sys

from bs4 import BeautifulSoup

input_folder = os.path.abspath(sys.argv[1])
output_file = os.path.abspath(sys.argv[2])

num_written = 0

with open(output_file, "w") as fp:
    dirs = os.listdir(input_folder)
    for f, file in enumerate(dirs):
        file_path = os.path.join(input_folder, file)
        content = open(file_path, "r").read()
        try:
            soup = BeautifulSoup(content, 'html.parser')
            title = soup.find("div", id="content").find("h1").text.strip()  # soup.find("title").text.strip()
            lang_nav = soup.find("nav", id="p-lang").find("div", {"class": "body vector-menu-content"}).find("ul")
            en_nav = lang_nav.find("li", {"class": "interlanguage-link interwiki-en"})
            en_link = en_nav.find("a")
            en_title, en_href = en_link["title"], en_link["href"]
            translation = en_href[en_href.find("wiki/") + 5:].strip().replace("_", " ")
            fp.write(title + "\t" + translation + "\n")
            num_written += 1
            if (f + 1) % 10 == 0:
                print(f + 1, "/", len(dirs), "-> wrote", num_written)  # , end="\r")
        except:
            pass
print("\nWrote", num_written)
