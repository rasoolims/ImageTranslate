import datetime
import errno
import os
import signal
import sys
import time
import urllib.request
from functools import wraps
from PIL import Image


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
def download_one_image(fixed_url, file_path):
    urllib.request.urlretrieve(fixed_url, file_path)


input_file = os.path.abspath(sys.argv[1])
output_folder = os.path.abspath(sys.argv[2])
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_indices = []

image_list = [line.strip().split("\t") for line in open(input_file, 'r').read().strip().split("\n") if
              len(line.strip().split("\t")) == 2]
print("loaded image_list", len(image_list))
file_number = 1
default_set = {"png", "jpg", "jpeg"}
url_count = 0
start_time = time.time()
file_path = os.path.join(output_folder, "index.txt")
with open(file_path, "w") as writer:
    for text, url in image_list:
        url_count += 1
        fixed_url = url
        if "?" in fixed_url:
            fixed_url = fixed_url[:fixed_url.find("?")]
        extension = "jpg"
        if extension not in default_set:
            continue
        else:
            file_extension = "." + extension

        file_path = os.path.join(output_folder, str(file_number) + file_extension)

        try:
            download_one_image(fixed_url, file_path)
            try:
                im = Image.open(os.path.abspath(file_path))
                x, y = im.size
                if x>=256 and y>=256:
                    new_im = im.resize((256, 256))
                    new_im.save(file_path)
                    file_indices.append(str(file_number) + "\t" + fixed_url + "\t" + text)
                    file_number += 1
            except:
                pass

        except:
            pass

        if url_count % 1 == 0:
            print(datetime.datetime.now(), url_count, "/", len(image_list), "->", (file_number-1), end="\r")
            start_time = time.time()
            if len(file_indices)>0:
                writer.write("\n".join(file_indices))
                writer.write("\n")
                file_indices = []

    sys.stdout.write(str(url_count) + "\n")
    writer.write("\n".join(file_indices))

print("\nWritten files", file_number)
