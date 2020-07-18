import os
from optparse import OptionParser
import os
import signal
import time
import urllib.error
import urllib.parse
import urllib.parse as urlparse
import urllib.request
import urllib.request
from functools import wraps
from optparse import OptionParser


class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


@timeout(300, "time out")
def download_one_file(fixed_url, file_path):
    urllib.request.urlretrieve(fixed_url, file_path)


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--file", dest="file", help="Which files to use", metavar="FILE", default=None)
    parser.add_option("--lang", dest="lang", help="Ref files to use for overlap", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_file", help="Output pickle file.", metavar="FILE", default=None)
    parser.add_option("--output_folder", dest="output_folder", help="Output pickle file.", metavar="FILE", default=None)
    parser.add_option("--resume", dest="resume", type="int", default=0)
    parser.add_option("--end", dest="end", type="int", default=100000000)
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    lang_id = options.lang
    url = "https://" + lang_id + ".wikipedia.org/wiki/"
    if not os.path.exists(options.output_folder):
        os.makedirs(options.output_folder)
    file_number = 0
    with open(options.file, "r") as fp, open(options.output_file, "w") as writer:
        for i, line in enumerate(fp):
            if i < options.resume or i >= options.end:
                continue
            title = line.strip().split("</s>")[0]
            title = title[title.find(">") + 1:].strip().replace(" ", "_")
            title_url = url + title
            writer.write(str(i) + "\t" + title + "\t" + title_url + "\n")

            parsed_link = urlparse.urlsplit(title_url)
            parsed_link = parsed_link._replace(path=urllib.parse.quote(parsed_link.path))
            fixed_url = parsed_link.geturl()

            output_path = os.path.join(options.output_folder, str(i) + ".html")

            total_tries = 2
            for t in range(total_tries):
                try:
                    download_one_file(fixed_url, output_path)
                    file_number += 1
                    break
                except:
                    if t == total_tries - 1:
                        print("\nunable to download\t" + output_path + "\t" + fixed_url+"\n")
                    time.sleep(1)
                    pass

            if (i + 1) % 10 == 0:
                print("Reading", i + 1, "Got", file_number, end="\r")

    print("\nFinished")
