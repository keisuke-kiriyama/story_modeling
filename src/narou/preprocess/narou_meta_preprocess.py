import os
import json
import re

from src.narou.corpus.narou_corpus import NarouCorpus
from src.util import settings
from narou_contents_preprocess import corpus
from narou_contents_preprocess import remove_escape_sequence
from narou_contents_preprocess import convert_serif_marker
from narou_contents_preprocess import preprocess_seq_serifs
from narou_contents_preprocess import splited_sentences

novel_meta_origin_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'meta_origin')
meta_origin_file_paths = [os.path.join(novel_meta_origin_dir_path, file_name) for file_name in os.listdir(novel_meta_origin_dir_path) if not file_name == '.DS_Store']

def preprocess_synopsis(synopsis):
    """
    あらすじの前処理
    :param synopsis: str
    :return: list
    """
    synopsis = remove_escape_sequence(synopsis)
    synopsis = convert_serif_marker(synopsis)
    synopsis_sentences = splited_sentences(synopsis)
    print(synopsis_sentences)

def preprocess_meta_data():
    file_path = meta_origin_file_paths[2]
    origin_data = corpus.load(file_path=file_path)
    synopsis = origin_data['story']
    processed_synopsis = preprocess_synopsis(synopsis)


if __name__ == '__main__':
    preprocess_meta_data()
