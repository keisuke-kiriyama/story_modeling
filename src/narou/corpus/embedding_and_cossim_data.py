import os
import joblib
import numpy as np

from src.narou.corpus.narou_corpus import NarouCorpus
from src.util import settings

class EmbeddingAndCosSimData:

    """
    X: 文中の単語ベクトルの平均ベクトル
    Y: 本文全文と最も類似度が高いあらすじ文との類似度
    {
        ncode:
            {
            error_line_indexes: np.array,
            X: np.array,
            Y: np.array
            }
        }
    }
    """

    def __init__(self):
        # Corpus
        self.corpus = NarouCorpus()
        self.non_seq_data_dict_emb_cossim_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'non_seq_data_dict_emb_cossim','non_seq_data_dict_emb_cossim.txt')

        # PROPERTY
        self.sentence_vector_size = 200

    def load_non_seq_data_dict_emb_cossim_data(self):
        """
        非系列情報の文ベクトルとコサイン類似度のを読み込む
        """
        print('loading tensor data...')
        with open(self.non_seq_data_dict_emb_cossim_path, 'rb') as f:
            data_dict = joblib.load(f)
        return data_dict

    def create_non_seq_tensors_emb_cossim_per_novel(self, ncode):
        """
        与えられたNコードの小説の文ベクトルとコサイン類似度のTensorを返却
        あらすじ文がない場合には空の(None, None)が返却される
        :param ncode: str
        :param is_test_data: bool
        :return: (np.array, np.array)
        """
        print('[PROCESS NCODE]: {}'.format(ncode))
        error_line_indexes = np.array([])
        X_per_novel = np.empty((0, self.sentence_vector_size), float)
        Y_per_novel = np.array([])
        contents_lines = self.corpus.get_contents_lines(ncode=ncode)
        synopsis_lines = self.corpus.get_synopsis_lines(ncode=ncode)
        if not synopsis_lines:
            return None, None, None
        contents_BoW_vectors, synopsis_BoW_vectors = self.corpus.get_BoW_vectors(contents_lines=contents_lines, synopsis_lines=synopsis_lines)
        for line_idx, (contents_line, contents_BoW_vector) in enumerate(zip(contents_lines, contents_BoW_vectors)):
            if line_idx % 30 == 0:
                print('{} progress: {:.1f}%'.format(ncode, line_idx / len(contents_lines) * 100))

            # 本文各文の文ベクトルをX_per_novelに追加
            try:
                sentence_vector = self.corpus.get_avg_word_vectors(contents_line)
                X_per_novel = np.append(X_per_novel, [sentence_vector], axis=0)
            except KeyError as err:
                print(err)
                error_line_indexes = np.append(error_line_indexes, line_idx)
                continue
            except:
                print('[Error] continue to add sentence vectors')
                error_line_indexes = np.append(error_line_indexes, line_idx)
                continue

            # 各文のあらすじ文との最大cos類似度をY_per_novelに追加
            max_sim = 0
            for synopsis_BoW_vector in synopsis_BoW_vectors:
                sim = self.corpus.cos_sim(contents_BoW_vector, synopsis_BoW_vector)
                if sim > max_sim:
                    max_sim = sim
            Y_per_novel = np.append(Y_per_novel, max_sim)

        return error_line_indexes, X_per_novel, Y_per_novel

    def create_non_seq_data_dict_emb_cossim(self):
        """
        非系列情報の文ベクトルとコサイン類似度のTensorを構築
        10ファイルごとにテンソルを保存
        :return: dict
        {
        ncode:
            {
            error_line_indexes: np.array,
            X: np.array,
            Y: np.array
            }
        }
        """
        data_dict = dict()
        for file_index, contents_file_path in enumerate(self.corpus.contents_file_paths):
            print('[INFO] num of processed novel count: {}'.format(file_index))
            ncode = self.corpus.ncode_from_contents_file_path(contents_file_path)
            error_line_indexes, X_per_novel, Y_per_novel = self.create_non_seq_tensors_emb_cossim_per_novel(ncode=ncode)
            if Y_per_novel is None:
                continue
            per_novel_dict = {'error_line_indexes': error_line_indexes, 'X': X_per_novel, 'Y': Y_per_novel}
            data_dict[ncode] = per_novel_dict

            # 100作品ごとにdictを保存する
            if file_index % 100 == 0:
                print('saving tensor...')
                with open(self.non_seq_data_dict_emb_cossim_path, 'wb') as f:
                    joblib.dump(data_dict, f, compress=3)
        print('saving tensor...')
        with open(self.non_seq_data_dict_emb_cossim_path, 'wb') as f:
            joblib.dump(data_dict, f, compress=3)
        return data_dict

    def non_seq_data_dict_emb_cossim(self, tensor_refresh=False):
        """
        非系列情報と仮定された文ベクトルとコサイン類似度のNコードをキーとする辞書を返却
        X. 全小説の全文をベクトルで表した２次元ベクトル
        - 文ベクトルは文中の単語ベクトルの平均ベクトル
        - shape = (小説数*文数, 文ベクトルサイズ)
        Y. 全小説の全文のコサイン類似度を要素とする１次元ベクトル
        :param tensor_refresh: bool
        :return: dict
        """
        is_tensor_data_exist = os.path.isfile(self.non_seq_data_dict_emb_cossim_path)
        if is_tensor_data_exist and not tensor_refresh:
            data_dict = self.load_non_seq_data_dict_emb_cossim_data()
        else:
            data_dict = self.create_non_seq_data_dict_emb_cossim()
        return data_dict

