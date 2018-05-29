# -*- coding: utf-8 -*-
import os
import csv
from src.util import settings

def extract_new_pseudonym_line(lines):
    necessary_meta_info_index = [0, 1, 2, 8, 9, 14, 15, 16, 17, 18, 45, 50]
    with open(os.path.join(settings.DATA_DIR_PATH, 'list_new_pseudonym.csv'), 'w') as list_new_pseudonym:
        writer = csv.writer(list_new_pseudonym, lineterminator='\n')
        for (i, line) in enumerate(lines):
            splited_line = line.replace('"', '').split(',')
            necessary_items = [splited_line[i] for i in necessary_meta_info_index]
            if i == 0 or (splited_line[9] == "新字新仮名" and splited_line[8] in ["NDC 913", "NDC K913"]):
                writer.writerow(necessary_items)

if __name__ == '__main__':
    list_person_all_extended_utf8_path = os.path.join(settings.DATA_DIR_PATH, 'list_person_all_extended_utf8.csv')
    with open(list_person_all_extended_utf8_path, 'r') as list_person_all:
        extract_new_pseudonym_line(list_person_all.readlines())
