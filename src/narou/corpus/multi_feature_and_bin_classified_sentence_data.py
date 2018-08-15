import os
import joblib
import numpy as np
import MeCab
import re

from src.narou.corpus.embedding_and_bin_classified_sentence_data import EmbeddingAndBinClassifiedSentenceData
from src.util import settings

class MultiFeatureAndBinClassifiedSentenceData:
    """
    EmbeddingAndBinClassifiedSentenceDataの入力ベクトルに
    様々な素性を加えたデータ
    error_line_indexes: データ作成時にエラーがでた文のインデックス
    X: 文中の単語ベクトルの平均ベクトル
    Y: 採用された文を正例、採用されなかった文を負例
    threshold: 最後に採用された文に付与されたスコア
    rouge: 最も値が高かった際のROUGEスコア
    {
        ncode:
         {
         error_line_indexes: np.array,
         X: np.array,
         y: np.array,
         threshold: float,
         rouge:
            {
            f: float
            r: float,
            p: float
            }
        }
    }
    加えた素性
    - 進度: float
    - 台詞か否か: 0,1
    - 固有名詞(人名)の数: int
    - 文字数: int
    """

    def __init__(self):

        # DATA
        self.base_data_supplier = EmbeddingAndBinClassifiedSentenceData()
        self.corpus = self.base_data_supplier.corpus

        # PATH
        self.multi_feature_and_bin_classified_sentence_data_dict_path = os.path.join(settings.NAROU_MODEL_DIR_PATH,
                                                                                     'multi_feature_and_bin_classified_sentence',
                                                                                     'multi_feature_and_bin_classified_sentence.txt')

        # PROPERTY
        self.input_vector_size = 204

    def load_data_dict(self):
        print('loading data dict...')
        with open(self.multi_feature_and_bin_classified_sentence_data_dict_path, 'rb') as f:
            data_dict = joblib.load(f)
        return data_dict

    def create_per_novel_data_dict(self, ncode, base_data):
        """
        各小説のデータを作成する
        """
        print('processing Ncode: {}'.format(ncode))
        m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

        contents_lines = self.corpus.get_contents_lines(ncode)
        removed_contents_lines = self.corpus.remove_error_line_indexes_from_contents_lines(contents_lines,
                                                                                           base_data['error_line_indexes'])
        input_tensor = np.empty((0, self.input_vector_size), float)
        for sentence_index, sentence_vector in enumerate(base_data['X']):
            if sentence_index % 30 == 0:
                print('{} progress: {:.1f}%'.format(ncode, sentence_index / len(removed_contents_lines) * 100))
            sentence = removed_contents_lines[sentence_index]

            morph_info = [re.split('[,\t]', morph) for morph in m.parse(sentence).split('\n')][:-2]
            # 進度
            story_progress = round(sentence_index / len(base_data['X']), 5)
            # 台詞か否か
            is_serif = int('「' in sentence)
            # 固有名詞の数
            person_name_count = len([morph for morph in morph_info if morph[3]=='人名'])
            # 文字数
            char_count = len(sentence)

            # 素性を追加した文のベクトル
            sentence_vector = np.append(sentence_vector, story_progress)
            sentence_vector = np.append(sentence_vector, is_serif)
            sentence_vector = np.append(sentence_vector, person_name_count)
            sentence_vector = np.append(sentence_vector, char_count)

            input_tensor = np.append(input_tensor, [sentence_vector], axis=0)

        return input_tensor

    def create_data_dict(self):
        """
        embedding_and_bin_classified_sentenceのデータの入力ベクトルに素性を追加して新たなデータを作成する
        :return: dict
        """
        base_data_dict = self.base_data_supplier.embedding_and_bin_classified_sentence_data_dict(data_refresh=False)
        data_dict = dict()
        for file_index, (ncode, data) in enumerate(base_data_dict.items()):
            print('[INFO] num of processed novel count: {}'.format(file_index))
            per_novel_dict = self.create_per_novel_data_dict(ncode=ncode, base_data=data)
            if per_novel_dict is None:
                continue
            data_dict[ncode] = per_novel_dict
        print('saving data_dict...')
        with open(self.multi_feature_and_bin_classified_sentence_data_dict_path, 'wb') as f:
            joblib.dump(data_dict, f, compress=3)
        return data_dict

    def multi_feature_and_bin_classified_sentence_data_dict(self, data_refresh=False):
        """
        データを新たに作るか、ロードするか判別し、データを返す
        {
         error_line_indexes: np.array,
         X: np.array,
         y: np.array,
         threshold: float,
         rouge:
            {
            f: float,
            r: float,
            p: float
            }
        }
        """
        is_data_dict_exist = os.path.isfile(self.multi_feature_and_bin_classified_sentence_data_dict_path)
        if is_data_dict_exist and not data_refresh:
            data_dict = self.load_data_dict()
        else:
            data_dict = self.create_data_dict()
        return data_dict

if __name__ == '__main__':
    supplier = MultiFeatureAndBinClassifiedSentenceData()
    supplier.multi_feature_and_bin_classified_sentence_data_dict(data_refresh=True)
