# -*- coding: utf-8 -*-

import os
import csv
from src.util import settings

def add_text_length(lines):
    new_pseudonym_texts_dir_path = os.path.join(settings.AOZORA_DATA_DIR_PATH, 'new_pseudonym_texts')
    output_file_path = os.path.join(settings.AOZORA_DATA_DIR_PATH, 'list_new_pseudonym_with_count.csv')
    output_file = open(output_file_path, 'w', encoding='shift_jisx0213')
    directories = os.listdir(new_pseudonym_texts_dir_path)
    for line in lines:
        id = str(line.split(',')[0])
        if id not in directories:
            continue
        text_dir_path = os.path.join(new_pseudonym_texts_dir_path, id)
        files = os.listdir(text_dir_path)
        text_files = [file for file in files if '.txt' in file]
        if len(text_files) == 0:
            continue
        text_file_path = os.path.join(text_dir_path, text_files[0])
        try:
            with open(text_file_path, 'r', encoding='shift-jis') as text_file:
                splited_line = line.strip().split(',')
                splited_line.append(len(text_file.read()))
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow(splited_line)
        except:
            print('error')
    output_file.close()

if __name__ == '__main__':
    list_new_pseudonym_path = os.path.join(settings.AOZORA_DATA_DIR_PATH, 'list_new_pseudonym.csv')
    with open(list_new_pseudonym_path, 'r') as list_new_pseudonym:
        lines = list_new_pseudonym.readlines()
        add_text_length(lines)
