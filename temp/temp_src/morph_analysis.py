# -*- coding: utf-8 -*-

import MeCab
import os
from src.util import settings

m = MeCab.Tagger()
temp_data_path = os.path.join(settings.TEMP_DIR_PATH, "temp_data/aohige.txt")

with open(temp_data_path, 'r', encoding='shift-jis') as temp_data:
    lines = temp_data.readlines()
    for line in lines:
        res = m.parseToNode(line)
        print(type(res.feature))
