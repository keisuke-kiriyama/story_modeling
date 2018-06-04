# -*- coding: utf-8 -*-

import re

def extract_body(text):
    text_parts = text.split('-' * 55)
    text_body = text_parts[2][0:text_parts[2].find('底本')].strip()
    return text_body

def remove_pseudonym_reading(text):
    text_body = extract_body(text)
    processed_body = re.sub('《.+?》|［.+?］|｜', '', text_body)
    return processed_body



