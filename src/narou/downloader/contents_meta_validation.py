# 小説本文とあらすじが両方存在していないファイルを削除する

from src.util import settings
from src.narou.downloader.narou_meta_downloader import fetch_novel_meta_info
import os
import json

def completion_meta():
    contents_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'contents')
    meta_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'meta')
    contents_file_names = os.listdir(contents_dir_path)
    meta_file_names = os.listdir(meta_dir_path)
    contents_n_codes = set([contents_file_name.split('.')[0] for contents_file_name in contents_file_names])
    meta_ncodes = set([meta_file_name.split('.')[0].replace('_meta', '') for meta_file_name in meta_file_names])
    n_code_diferrences = contents_n_codes.difference(meta_ncodes)
    print(n_code_diferrences)
    for none_meta_ncode in n_code_diferrences:
        if not none_meta_ncode: continue
        meta = fetch_novel_meta_info(none_meta_ncode)
        output_file_path = os.path.join(meta_dir_path, none_meta_ncode + '_meta.json')
        with open(output_file_path, 'w') as f:
            json.dump(meta, f, ensure_ascii=False)
            print(meta)

if __name__ == '__main__':
    completion_meta()
