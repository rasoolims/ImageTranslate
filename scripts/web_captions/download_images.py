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
start_index = int(sys.argv[2])
end_index = int(sys.argv[3])
output_folder = os.path.abspath(sys.argv[4])
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_indices = []

file_number = 1 + start_index
default_set = {"png", "jpg", "jpeg"}
url_count = 0
start_time = time.time()
file_path = os.path.join(output_folder, "index." + str(start_index) + ".txt")
with open(file_path, "w") as writer, open(input_file, 'r') as reader:
    for line in reader:
        try:
            url_count += 1
            if start_index > url_count or url_count > end_index:
                continue
            text, url = line.strip().split("\t")
            fixed_url = url
            if "?" in fixed_url:
                fixed_url = fixed_url[:fixed_url.find("?")]
            extension = fixed_url[fixed_url.rfind(".") + 1:].lower()
            if extension not in default_set:
                continue
            else:
                file_extension = "." + extension

            file_path = os.path.join(output_folder, str(file_number) + file_extension)

            download_one_image(fixed_url, file_path)
            try:
                im = Image.open(os.path.abspath(file_path))
                x, y = im.size
                if x >= 256 or y >= 256:
                    new_im = im.resize((256, 256))
                    new_im.save(file_path)
                    file_indices.append(str(file_number) + file_extension + "\t" + fixed_url + "\t" + text)
                    file_number += 1
            except:
                pass

        except:
            pass

        if url_count % 1 == 0:
            print(datetime.datetime.now(), url_count, "->", (file_number - 1), end="\r")
            start_time = time.time()
            if len(file_indices) > 0:
                writer.write("\n".join(file_indices))
                writer.write("\n")
                file_indices = []

    sys.stdout.write(str(url_count) + "\n")
    writer.write("\n".join(file_indices))

print("\nWritten files", file_number)
