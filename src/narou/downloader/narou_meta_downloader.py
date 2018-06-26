import os
import json
import gzip
from urllib.request import urlopen, HTTPError, URLError
from time import sleep

from src.util import settings

def fetch_novel_meta_info(n_code, error_log_file):
    url = 'https://api.syosetu.com/novelapi/api/?out=json&gzip=5&of=t-n-u-w-s-bg-g-k-nt-e-ga-l-gp-f-r-a-ah-ka-&lim=1&ncode={}'.format(n_code)
    sleep(3)
    print("fetch: {}".format(url))
    response = urlopen(url)
    try:
        with gzip.open(response, 'rt', encoding='utf-8') as f:
            j_raw = f.read()
            j_obj = json.loads(j_raw)
            novel_meta = j_obj[1]
    except:
        error_log_file.write(n_code)
    return novel_meta

def fetch_novel_meta_from_ncode(ncode_file_path, output_file_path, error_log_file_path):
    ncode_file = open(ncode_file_path, 'r')
    output_file = open(output_file_path, 'w')
    error_log_file = open(error_log_file_path, 'w')
    lines = ncode_file.readlines()
    for line in lines:
        n_code = line.split(',')[1].replace('"', '').strip()
        meta = fetch_novel_meta_info(n_code, error_log_file)
        json.dump(meta, output_file)
    ncode_file.close()
    output_file.close()
    error_log_file.close()

if __name__ == '__main__':
    csv_file_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'items_narou_ncode_spider_10.csv')
    output_file_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'novel_meta.json')
    error_log_file_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'fetch_error_log.txt')
    fetch_novel_meta_from_ncode(csv_file_path, output_file_path, error_log_file_path)



