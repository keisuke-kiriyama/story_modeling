# -*- coding: utf-8 -*-

from os import path, listdir
import shutil
import re
from src.util import settings

def extract_literature_from_PB():
    """
    story_modeling/data/bccwj/PBにあるファイルのうち
    PB●9で始まるテキストをstory_modeling/data/bccwj/literatureに移動する
    """
    pb_dir_path = path.join(settings.BCCWJ_DATA_DIR_PATH, 'PB')
    pb_sub_dir_paths = list(map(lambda x: path.abspath(path.join(pb_dir_path, x)), listdir(pb_dir_path)))
    literature_dir_path = path.join(settings.BCCWJ_DATA_DIR_PATH, 'Literature')
    for pb_sub_dir_path in pb_sub_dir_paths:
        if path.basename(pb_sub_dir_path) == '.DS_Store': continue
        file_names = listdir(pb_sub_dir_path)
        for file_name in file_names:
            is_literature = re.match('^PB\d9', file_name)
            if is_literature:
                shutil.move(path.join(pb_sub_dir_path, file_name), literature_dir_path)


if __name__ == '__main__':
    extract_literature_from_PB()
