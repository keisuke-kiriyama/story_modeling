# -*- coding: utf-8 -*-

import os
from src.util import settings
from src.downloader.text_downloader import download_zip

def download_new_pseudonym_text(lines):
    for line in lines:
        splited_line = line.split(',')
        id = splited_line[0]
        url = splited_line[9]
        output_file_path = os.path.join(settings.DATA_DIR_PATH, 'new_pseudonym_texts', str(id))
        download_zip(url, output_file_path)

if __name__ == '__main__':
    list_new_pseudonym_path = os.path.join(settings.DATA_DIR_PATH, 'list_new_pseudonym.csv')
    with open(list_new_pseudonym_path, 'r') as list_new_pseudonym:
        lines = list_new_pseudonym.readlines()
        download_new_pseudonym_text(lines[1:])
