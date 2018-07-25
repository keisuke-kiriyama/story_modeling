import os
import json
import re

from src.narou.corpus.narou_corpus import NarouCorpus
from src.util import settings

novel_contents_origin_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'contents_origin')
contents_origin_file_paths = [os.path.join(novel_contents_origin_dir_path, file_name) for file_name in os.listdir(novel_contents_origin_dir_path) if not file_name == '.DS_Store']

# メソッドを用いる
corpus = NarouCorpus()

def remove_escape_sequence(contents):
    contents = contents.replace('\u3000', '')
    contents = contents.replace('\n', '')
    contents = contents.replace(' ', '')
    return contents

def convert_serif_marker(contents):
    contents = contents.replace('『', '「')
    contents = contents.replace('』', '」')
    return contents

def preprocess_seq_serifs(seq_serif):
    """
    連続したセリフを分割する
    """
    splited_serifs = re.split('(」)(?=「)', seq_serif)
    for idx, elem in enumerate(splited_serifs):
        splited_serifs[idx-1:idx+1] = [''.join(splited_serifs[idx-1:idx+1])]
    return splited_serifs

def splited_sentences(contents):
    """
    文単位で分割する
    句点で分割
    ただし、セリフ内の分割は行わない
    連続するセリフはセリフ間で分割する
    文内にあるセリフはセリフごと分割することをしない
    """
    anchor = '「'
    contents = contents + anchor
    delimiter = '。？！'
    pattern = '([' + delimiter + '])(?=[^」]*「)'
    prog = re.compile(pattern)
    splited = re.split(prog, contents)[:-1]
    sentences = []
    for idx, elem in enumerate(splited):
        if elem in delimiter:
            delimiter_joined_sentence = ''.join(splited[idx-1:idx+1])
            if '」「' in delimiter_joined_sentence:
                serifs = preprocess_seq_serifs(delimiter_joined_sentence)
                sentences.extend(serifs)
            else:
                sentences.append(delimiter_joined_sentence)
    return sentences

def preprocess_one_episode(contents_lines):
    """
    本文1話の前処理
    全文連結し、エスケープシーケンス等を削除
    その後文分割を行う
    :param contents_lines: list
    :return: list
    """
    contents = ''.join(contents_lines)
    contents = remove_escape_sequence(contents)
    contents = convert_serif_marker(contents)
    sentences = splited_sentences(contents)
    return sentences

def preprocess_contents_data():
    """
    スクレイピングしたデータを正確に文分割したデータに修正する
    スクレイピングしたデータをcontents_originにまとめ、この関数を回す
    前処理されたファイルはcontentsディレクトリ下に保存される
    :return:
    """
    for i, contents_origin_file_path in enumerate(contents_origin_file_paths):
        ncode = corpus.ncode_from_contents_file_path(contents_origin_file_path)
        print('progress: {:.1f}%, processing: {}'.format(i / len(contents_origin_file_paths) * 100, ncode))
        origin_data = corpus.load(file_path=contents_origin_file_path)
        data = dict()
        data['n_code'] = origin_data['n_code']
        data['sub_titles'] = origin_data['sub_titles']
        data['contents'] = []
        for origin_contents in origin_data['contents']:
            processed_episode = preprocess_one_episode(origin_contents)
            data['contents'].append(processed_episode)
        output_file_path = corpus.create_contents_file_path(ncode=ncode)
        with open(output_file_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False)


if __name__ == '__main__':
    preprocess_contents_data()


