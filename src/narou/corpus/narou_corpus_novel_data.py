import os
import joblib
import numpy as np
import MeCab
import re
from collections import Counter

from src.narou.corpus.narou_corpus import NarouCorpus
from src.util import settings

class NarouCorpusNovelData:
    """
    各小説の様々なデータを構築する
    """

    def __init__(self):

        # CORPUS
        self.corpus = NarouCorpus()

        # PATH
        self.narou_corpus_novel_data_path = os.path.join(settings.NAROU_MODEL_DIR_PATH,
                                                         'narou_corpus_novel_data',
                                                         'narou_corpus_novel_data.txt')
        self.ncodes_data_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'ncode.jl')

        # TAGGER
        self.tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

    def load_novel_data(self):
        print('loading novel data...')
        with open(self.narou_corpus_novel_data_path, 'rb') as f:
            data_dict = joblib.load(f)
        return data_dict

    def morph_distribution(self, morph_info, morph_info_index, target):
        """
        単語の出現分布の辞書を返す
        :param morph_info_index: int
        形態素解析した情報のリストの何番目のインデックスを見るか
        :param target: str
        一致する文字列
        :return: dict
        """
        morphs = [morph[0] for morph in morph_info if morph[morph_info_index] == target]
        distribution_dict = dict()
        counter = Counter(morphs)
        for word, cnt in counter.most_common():
            distribution_dict[word] = cnt
        return distribution_dict

    def create_per_novel_data(self, ncode):
        """
        ここの小説のデータ
        :param ncode: str
        :return: dict
        {
        proper_noun_distribution:
            { word: count }
        noun_distribution:
            { word: count }
        person_distribution:
            { word: count }
        person_distribution:
                { word: count }
        verb_distribution:
            { word: count }
        }
        max_sentence_lentgh: int
        """
        contents_lines = self.corpus.get_contents_lines(ncode)
        morph_info = self.corpus.get_morph_info(contents_lines)

        # 固有名詞の出現分布
        proper_noun_distribution = self.morph_distribution(morph_info=morph_info, morph_info_index=2, target='固有名詞')

        # 名詞の出現分布
        noun_distribution = self.morph_distribution(morph_info=morph_info, morph_info_index=1, target='名詞')

        # 人名の出現分布
        person_distribution = self.morph_distribution(morph_info=morph_info, morph_info_index=3, target='人名')

        # 地名の出現分布
        place_distribution = self.morph_distribution(morph_info=morph_info, morph_info_index=3, target='地域')

        # 動詞の出現分布
        verb_distribution = self.morph_distribution(morph_info=morph_info, morph_info_index=1, target='動詞')

        # 文の最大文字数
        max_sentence_length = max([len(line) for line in contents_lines])

        # 辞書の作成
        data = {
            'proper_noun_distribution': proper_noun_distribution,
            'noun_distribution': noun_distribution,
            'person_distribution': person_distribution,
            'place_distribution': place_distribution,
            'verb_distribution': verb_distribution,
            'max_sentence_length': max_sentence_length
        }

        return data

    def create_novel_data(self):
        with open(self.ncodes_data_path, 'r') as f:
            self.ncodes = [line.split('"')[3] for line in f.read().split('\n')[:-1]]
        novel_data_dict = dict()
        for file_index, ncode in enumerate(self.ncodes):
            print('[INFO] num of processed novel count: {}'.format(file_index))
            print('processing Ncode: {}'.format(ncode))
            if not self.corpus.create_contents_file_path(ncode) in self.corpus.contents_file_paths: continue
            per_novel_data = self.create_per_novel_data(ncode)
            novel_data_dict[ncode] = per_novel_data
        print('saving novel data...')
        with open(self.narou_corpus_novel_data_path, 'wb') as f:
            joblib.dump(novel_data_dict, f, compress=3)
        return novel_data_dict

    def narou_corpus_novel_data(self, data_refresh=False):
        """
        データを新たに作るか、ロードするか判別し、データを返す
        {
            ncode:
            {
                proper_noun_distribution:
                    { word: count }
                noun_distribution:
                    { word: count }
                person_distribution:
                    { word: count }
                place_distribution:
                    { word: count }
                verb_distribution:
                    { word: count }
                max_sentence_lentgh: int
                }
            }
        }
        """
        is_data_exist = os.path.isfile(self.narou_corpus_novel_data_path)
        if is_data_exist and not data_refresh:
            novel_data = self.load_novel_data()
        else:
            novel_data = self.create_novel_data()
        return novel_data


if __name__ == '__main__':
    supplier = NarouCorpusNovelData()
    # supplier.narou_corpus_novel_data(True)
    print(supplier.create_per_novel_data('n1185df'))