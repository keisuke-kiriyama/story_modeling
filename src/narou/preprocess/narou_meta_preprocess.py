import os
import json
import re

from src.narou.corpus.narou_corpus import NarouCorpus
from src.util import settings
from narou_contents_preprocess import corpus
from narou_contents_preprocess import remove_escape_sequence
from narou_contents_preprocess import convert_serif_marker
from narou_contents_preprocess import splited_sentences

novel_meta_origin_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'meta_origin')
meta_origin_file_paths = [os.path.join(novel_meta_origin_dir_path, file_name) for file_name in os.listdir(novel_meta_origin_dir_path) if not file_name == '.DS_Store']

def remove_publicity_sentences(sentences):
    """
    宣伝用の文を除去する
    """
    pablicity_words = [
        'ランキング',
        '日間',
        '月間',
        '累計',
        'アルファポリス',
        '完結済',
        'ハイファンタジー',
        'PV',
        '休載',
        'http',
        '本編',
        '番外編',
        'フィクション',
        '携帯版',
        '投稿予定',
        'サブタイ'
    ]
    removed_sentences = []
    for sentence in sentences:
        is_publicity_sentence = True in [pablicity_word in sentence for pablicity_word in pablicity_words]
        if not is_publicity_sentence and len(sentence) > 1:
            removed_sentences.append(sentence)
    return removed_sentences

def preprocess_synopsis(synopsis):
    """
    あらすじの前処理
    :param synopsis: str
    :return: list
    """
    if not synopsis[-1] in ['。', '？', '！']:
        synopsis += '。'
    synopsis = remove_escape_sequence(synopsis)
    synopsis = convert_serif_marker(synopsis)
    synopsis_sentences = splited_sentences(synopsis)
    preprocessed_sentences = remove_publicity_sentences(synopsis_sentences)
    return preprocessed_sentences

def preprocess_meta_data():
    """
    あらすじの文分割を行い、宣伝用の文などを除去する
    スクレイピングしたデータをmeta_originにまとめ、この関数を回す
    前処理されたファイルはmetaディレクトリ下に保存される
    """
    for i, meta_origin_file_path in enumerate(meta_origin_file_paths):
        ncode = corpus.ncode_from_meta_file_path(meta_origin_file_path)
        print('progress: {:.1f}%, processing: {}'.format(i / len(meta_origin_file_paths) * 100, ncode))
        meta_data = corpus.load(file_path=meta_origin_file_path)
        meta_data['story'] = preprocess_synopsis(meta_data['story'])
        output_file_path = corpus.create_meta_file_path(ncode=ncode)
        with open(output_file_path, 'w') as f:
            json.dump(meta_data, f, ensure_ascii=False)

if __name__ == '__main__':
    preprocess_meta_data()
