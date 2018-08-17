import os
import joblib
import numpy as np
import MeCab
import re
import copy

from src.narou.corpus.embedding_and_bin_classified_sentence_data import EmbeddingAndBinClassifiedSentenceData
from src.util import settings

class MultiFeatureAndBinClassifiedSentenceData:
    """
    EmbeddingAndBinClassifiedSentenceDataの入力ベクトルに
    様々な素性を加えたデータ
    error_line_indexes: データ作成時にエラーがでた文のインデックス
    X: 文中の単語ベクトルの平均ベクトル
    Y: 採用された文を正例、採用されなかった文を負例
    Y_score: 本文とあらすじ文との類似度が最も高い値
    threshold: 最後に採用された文に付与されたスコア
    rouge: 最も値が高かった際のROUGEスコア
    {
        ncode:
         {
         error_line_indexes: np.array,
         X: np.array,
         Y: np.array,
         Y_score: np.array,
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

        # TAGGER
        self.tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

        # PROPERTY
        self.input_vector_size = 204
        self.new_features_count = 4

    def load_data_dict(self):
        print('loading data dict...')
        with open(self.multi_feature_and_bin_classified_sentence_data_dict_path, 'rb') as f:
            data_dict = joblib.load(f)
        return data_dict

    def create_multi_feature_input_vector(self, sentence, sentence_index, all_sentence_count, base_input_vector):
        """
        文のベクトルに複数の素性を追加したベクトルを返す
        :param sentence: str
        :param base_input_vector: np.array
        :return: np.array
        """
        morph_info = [re.split('[,\t]', morph) for morph in self.tagger.parse(sentence).split('\n') if morph not in ['', 'EOS']]

        # 進度
        story_progress = round(sentence_index / all_sentence_count, 5)
        # 台詞か否か
        is_serif = int('「' in sentence)
        # 固有名詞の数
        person_name_count = len([morph for morph in morph_info if morph[3]=='人名'])
        # 文字数
        char_count = len(sentence)

        # 新たな素性一覧
        features = np.array([story_progress, is_serif, person_name_count, char_count])

        # 素性を追加した文のベクトル
        update_vector = np.append(base_input_vector, features)
        return update_vector

    def create_data_dict(self):
        """
        embedding_and_bin_classified_sentenceのデータの入力ベクトルに素性を追加して新たなデータを作成する
        :return: dict
        """
        base_data_dict = self.base_data_supplier.embedding_and_bin_classified_sentence_data_dict(data_refresh=False)
        data_dict = copy.deepcopy(base_data_dict)
        for file_index, (ncode, base_data) in enumerate(base_data_dict.items()):
            data_dict[ncode]['X'] = np.zeros((base_data['X'].shape[0], base_data['X'].shape[1] + self.new_features_count), float)
            print('[INFO] num of processed novel count: {}'.format(file_index))
            print('processing Ncode: {}'.format(ncode))

            contents_lines = self.corpus.get_contents_lines(ncode)
            removed_contents_lines = self.corpus.remove_error_line_indexes_from_contents_lines(contents_lines,
                                                                                               base_data['error_line_indexes'])
            for sentence_index, sentence_vector in enumerate(base_data['X']):
                if sentence_index % 30 == 0:
                    print('{} progress: {:.1f}%'.format(ncode, sentence_index / len(removed_contents_lines) * 100))
                sentence = removed_contents_lines[sentence_index]
                input_vector = self.create_multi_feature_input_vector(sentence=sentence,
                                                                      sentence_index=sentence_index,
                                                                      all_sentence_count=len(removed_contents_lines),
                                                                      base_input_vector=sentence_vector)
                data_dict[ncode]['X'][sentence_index] = input_vector
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
         Y: np.array,
         Y_score: np.array,
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
    supplier.create_data_dict()
