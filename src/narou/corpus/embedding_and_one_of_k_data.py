import os
import joblib
import numpy as np

from src.narou.corpus.narou_corpus import NarouCorpus
from src.util import settings

class EmbeddingAndOneOfKData:

    """
    X: 文中の単語ベクトルの平均ベクトル
    Y: 本文中の文が各あらすじ文と最も類似しているかをOneHotVectorで表現
    """

    def __init__(self):
        # Corpus
        self.corpus = NarouCorpus()
        self.non_seq_data_dict_emb_one_of_k_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'non_seq_data_dict_emb_one_of_k', 'non_seq_data_dict_emb_one_of_k.txt')

        # PROPERTY
        self.sentence_vector_size = 200

    def load_non_seq_data_dict_emb_one_of_k(self):
        """
        非系列情報の文ベクトルとあらすじ文と対応しているか否かのOneHotVectorの辞書を読み込む
        """
        print('loading tensor data...')
        with open(self.non_seq_data_dict_emb_one_of_k_path, 'rb') as f:
            data_dict = joblib.load(f)
        return data_dict

    def create_non_seq_tensors_emb_one_of_k_per_novel(self, ncode):
        """
        与えられたNコードの小説の文ベクトルと本文各文があらすじ文に対応しているかを表現するOneHotVectorを返却
        あらすじが空の場合は(None, None)が返される
        :param ncode: str
        :param exist_X: np.array
        :return: (np.array, np.array)
        """
        print('[PROCESS NCODE]: {}'.format(ncode))
        X_per_novel = np.empty((0, self.sentence_vector_size), float)
        contents_lines = self.corpus.get_contents_lines(ncode=ncode)
        synopsis_lines = self.corpus.get_synopsis_lines(ncode=ncode)
        if not synopsis_lines or not contents_lines:
            return None, None
        contents_BoW_vectors, synopsis_BoW_vectors = self.corpus.get_BoW_vectors(contents_lines=contents_lines, synopsis_lines=synopsis_lines)
        error_line_indexes = []
        for line_idx, contents_line in enumerate(contents_lines):
            if line_idx % 30 == 0:
                print('{} progress X: {:.1f}%'.format(ncode, line_idx / len(contents_lines) * 100))
            # 本文各文の文ベクトルをX_per_novelに追加
            try:
                sentence_vector = self.corpus.get_avg_word_vectors(contents_line)
                X_per_novel = np.append(X_per_novel, [sentence_vector], axis=0)
            except KeyError as err:
                print(err)
                error_line_indexes.append(line_idx)
                continue
            except:
                print('[Error] continue to add sentence vectors')
                error_line_indexes.append(line_idx)
                continue
        Y_per_novel = np.zeros(len(X_per_novel))
        # 例外が派生した文のBoWを削除する
        for error_line_index in sorted(error_line_indexes, reverse=True):
            del contents_BoW_vectors[error_line_index]
        for line_idx, synopsis_BoW_vector in enumerate(synopsis_BoW_vectors):
            print('{} progress Y: {:.1f}%'.format(ncode, line_idx / len(synopsis_lines) * 100))
            similarity = [self.corpus.cos_sim(contents_BoW_vector, synopsis_BoW_vector) for contents_BoW_vector in contents_BoW_vectors]
            max_similarity_index = np.argmax(similarity)
            Y_per_novel[max_similarity_index] = 1
        return X_per_novel, Y_per_novel


    def create_non_seq_data_dict_emb_one_of_k(self):
        """
        文ベクトルと各文があらすじ文に対応しているかをOneHotVectorで表したNコードをキーとする辞書を作成
        :param reuse_emb_cossim_X: bool
        :return: dict
        {
        ncode:
            {
            X: np.array,
            Y: np.array
            }
        }
        """
        data_dict = dict()
        for file_index, contents_file_path in enumerate(self.corpus.contents_file_paths):
            print('[INFO] num of processed novel count: {}'.format(file_index))
            ncode = self.corpus.ncode_from_contents_file_path(contents_file_path)
            X_per_novel, Y_per_novel = self.create_non_seq_tensors_emb_one_of_k_per_novel(ncode=ncode)
            if Y_per_novel is None:
                continue
            per_novel_dict = {'X': X_per_novel, 'Y': Y_per_novel}
            data_dict[ncode] = per_novel_dict
            # 10作品ごとにdictを保存する
            if file_index % 10 == 0:
                print('saving tensor...')
                with open(self.non_seq_data_dict_emb_one_of_k_path, 'wb') as f:
                    joblib.dump(data_dict, f, compress=3)
        print('saving tensor...')
        with open(self.non_seq_data_dict_emb_one_of_k_path, 'wb') as f:
            joblib.dump(data_dict, f, compress=3)
        return data_dict


    def non_seq_data_dict_emb_one_of_k(self, tensor_refresh=False):
        """
        文ベクトルと各文があらすじ文に対応しているかをOneHotVectorで表したNコードをキーとする辞書を返却
        :param tensor_refresh: tensor_refresh=Bool
        :return: dict
        {
        ncode:
            {
            X: np.array,
            Y: np.array
            }
        }
        """
        is_tensor_data_exist = os.path.isfile(self.non_seq_data_dict_emb_one_of_k_path)
        if is_tensor_data_exist and not tensor_refresh:
            data_dict = self.load_non_seq_data_dict_emb_one_of_k()
        else:
            data_dict = self.create_non_seq_data_dict_emb_one_of_k()
        return data_dict

