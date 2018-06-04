# -*- coding: utf-8 -*-

import os
import re
from src.util import settings

def extract_body(file_path):
    file = open(file_path, 'r', encoding='shift-jis')
    text_parts = file.read().split('-' * 55)
    text_body = text_parts[2][0:text_parts[2].find('底本')].strip()
    file.close()
    return text_body

def remove_pseudonym_reading(text_body):
    processed_body = re.sub('《.+?》|［.+?］|｜', '', text_body)
    return processed_body

if __name__ == '__main__':
    # file_path = os.path.join(settings.TEMP_DATA_PATH, 'neboke.txt')
    file_path = os.path.join(settings.TEMP_DATA_PATH, 'aohige.txt')
    text_body = extract_body(file_path)
    processed_body = remove_pseudonym_reading(text_body)
    print(processed_body)

