import os
import json
import re
import numpy as np
import MeCab
from itertools import chain
import joblib
from gensim.models import word2vec
from gensim import corpora, matutils

from src.util import settings

class NarouCorpus:

    def __init__(self):
        # PATHS
        self.novel_contents_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'contents')
        self.novel_meta_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'meta')
        self.contents_file_paths = [os.path.join(self.novel_contents_dir_path, file_name) for file_name in os.listdir(self.novel_contents_dir_path) if not file_name == '.DS_Store']
        self.meta_file_paths = [os.path.join(self.novel_meta_dir_path, file_name) for file_name in os.listdir(self.novel_meta_dir_path) if not file_name == '.DS_Store']
        self.non_seq_data_dict_emb_cossim_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'non_seq_data_dict_emb_cossim','non_seq_data_dict_emb_cossim.txt')
        self.non_seq_data_dict_emb_cossim_train_ncode_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'non_seq_data_dict_emb_cossim','non_seq_data_dict_emb_cossim_train_ncode.txt')
        self.non_seq_data_dict_emb_cossim_test_ncode_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'non_seq_data_dict_emb_cossim','non_seq_data_dict_emb_cossim_test_ncode.txt')
        self.non_seq_data_dict_emb_one_of_k_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'non_seq_data_dict_emb_one_of_k', 'non_seq_data_dict_emb_one_of_k.txt')

        # MODELS
        self.word_embedding_model = self.load_embedding_model()

        # PROPERTY
        self.sentence_vector_size = 200

    def load_embedding_model(self):
        print('loading embedding_model...')
        embedding_model_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'narou_embedding.model')
        return word2vec.Word2Vec.load(embedding_model_path)

    def ncode_from_contents_file_path(self, file_path):
        return file_path.split('/')[-1].split('.')[0]

    def ncode_from_meta_file_path(self, file_path):
        return file_path.split('/')[-1].split('_')[0]

    def load(self, file_path):
        json_file = open(file_path, 'r')
        data = json.load(json_file)
        json_file.close()
        return data

    def wakati(self, line):
        m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati')
        wakati = m.parse(line).replace('\n', '')
        return wakati

    def create_contents_file_path(self, ncode):
        dir_path = self.novel_contents_dir_path
        return os.path.join(dir_path, ncode + '.json')

    def create_meta_file_path(self, ncode):
        dir_path = self.novel_meta_dir_path
        return os.path.join(dir_path, ncode+'_meta.json')

    def cleaning(self, line):
        line = line.replace('\u3000', '')
        line = line.replace('\n', '')
        line = line.replace(' ', '')
        return line

    def cos_sim(self, v1, v2):
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def get_contents_lines(self, ncode):
        """
        本文全文のリストを返却
        :param contents_file_path: str
        :return: list
        """
        contents_file_path = self.create_contents_file_path(ncode=ncode)
        if not contents_file_path in self.contents_file_paths:
            print('nothing ncode')
            return
        return list(chain.from_iterable(self.load(contents_file_path)['contents']))

    def get_synopsis_lines(self, ncode):
        """
        あらすじの文のリストを返却
        :param synopsis_file_path: str
        :return: list
        """
        meta_file_path = self.create_meta_file_path(ncode=ncode)
        if not meta_file_path in self.meta_file_paths:
            print('nothing ncode')
            return
        return self.load(meta_file_path)['story']

    def max_sentences_count(self):
        """
        全小説から最も文数の多い小説の文数を取得
        :return: int
        """
        sentences_count = []
        for file_path in self.contents_file_paths:
            ncode = self.ncode_from_contents_file_path(file_path)
            contents_lines = self.get_contents_lines(ncode=ncode)
            sentences_count.append(len(contents_lines))
        return max(sentences_count)

    def get_wakati_lines(self, lines):
        """
        文のリストを、分かち書きが行われた文のリストに変換
        :param lines: list
        :return: list
        """
        return [self.wakati(line).split() for line in lines]

    def get_wakati_contents_lines(self, ncode):
        """
        本文の各文を分かち書きしたリストを取得
        :param ncode: str
        :return: list
        """
        contents_lines = self.get_contents_lines(ncode=ncode)
        wakati_contents_lines = self.get_wakati_lines(contents_lines)
        return wakati_contents_lines

    def get_wakati_synopsis_lines(self, ncode):
        """
        あらすじの各分を分かち書きしたリストを取得
        :param ncode:
        :return:
        """
        synopsis_lines = self.get_synopsis_lines(ncode=ncode)
        wakati_synopsis_lines = self.get_wakati_lines(synopsis_lines)
        return wakati_synopsis_lines

    def remove_stop_word(self, sentence):
        """
        文中の名詞、形容詞、動詞、副詞のリストを返却
        :param sentence: str
        :return: list
        """
        part = ['名詞', '動詞', '形容詞', '副詞']
        m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        morphs = m.parse(sentence).split('\n')
        removed = []
        for morph in morphs:
            splited = re.split('[,\t]', morph)
            if len(splited) < 2: continue
            if splited[1] in part:
                removed.append(splited[0])
        return removed

    def get_BoW_vectors(self, contents_lines, synopsis_lines):
        """
        文のリストから各文のBoWベクトルのリストを返す
        :param contents_lines: list
        本文の各文を要素とするリスト
        :param synopsis_lines: list
        あらすじの各文を要素とするリスト
        :return: ([np.array], [np.array])
        """
        print('creating BoW vectors...')
        removed_contents_lines = [self.remove_stop_word(self.cleaning(line)) for line in contents_lines]
        removed_synopsis_lines = [self.remove_stop_word(self.cleaning(line)) for line in synopsis_lines]
        all_lines = removed_contents_lines + removed_synopsis_lines
        vocaburaly = corpora.Dictionary(all_lines)
        contents_BoWs = [vocaburaly.doc2bow(line) for line in removed_contents_lines]
        synopsis_BoWs = [vocaburaly.doc2bow(line) for line in removed_synopsis_lines]
        contents_vectors = [np.array(matutils.corpus2dense([bow], num_terms=len(vocaburaly)).T[0]) for bow in contents_BoWs]
        synopsis_vectors = [np.array(matutils.corpus2dense([bow], num_terms=len(vocaburaly)).T[0]) for bow in synopsis_BoWs]
        return contents_vectors, synopsis_vectors

    def get_avg_word_vectors(self, sentence):
        """
        文中の各単語の平均ベクトル返却
        :param sentence: str
        :return: np.array
        """
        wakati_sentence = self.wakati(sentence).split()
        word_vectors = np.array([self.word_embedding_model.__dict__['wv'][word] for word in wakati_sentence])
        return np.average(word_vectors, axis=0)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    X: 文中の単語ベクトルの平均ベクトル
    Y: 本文全文と最も類似度が高いあらすじ文との類似度
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""

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
        lines_per_novel = np.array([])
        X_per_novel = np.empty((0, self.sentence_vector_size), float)
        Y_per_novel = np.array([])
        contents_lines = self.get_contents_lines(ncode=ncode)
        synopsis_lines = self.get_synopsis_lines(ncode=ncode)
        if not synopsis_lines:
            return None, None
        contents_BoW_vectors, synopsis_BoW_vectors = self.get_BoW_vectors(contents_lines=contents_lines, synopsis_lines=synopsis_lines)
        for line_idx, (contents_line, contents_BoW_vector) in enumerate(zip(contents_lines, contents_BoW_vectors)):
            if line_idx % 30 == 0:
                print('{} progress: {:.1f}%'.format(ncode, line_idx / len(contents_lines) * 100))

            # 本文各文の文ベクトルをX_per_novelに追加
            try:
                sentence_vector = self.get_avg_word_vectors(contents_line)
                X_per_novel = np.append(X_per_novel, [sentence_vector], axis=0)
            except KeyError as err:
                print(err)
                continue
            except:
                print('[Error] continue to add sentence vectors')
                continue

            # 各文のあらすじ文との最大cos類似度をY_per_novelに追加
            max_sim = 0
            for synopsis_BoW_vector in synopsis_BoW_vectors:
                sim = self.cos_sim(contents_BoW_vector, synopsis_BoW_vector)
                if sim > max_sim:
                    max_sim = sim
            Y_per_novel = np.append(Y_per_novel, max_sim)

            # 処理済みの文を保持
            lines_per_novel = np.append(lines_per_novel, contents_lines[line_idx])

        return lines_per_novel, X_per_novel, Y_per_novel

    def create_non_seq_data_dict_emb_cossim(self):
        """
        非系列情報の文ベクトルとコサイン類似度のTensorを構築
        10ファイルごとにテンソルを保存
        :return: dict
        {
        ncode:
            {
            lines: np.array,
            X: np.array,
            Y: np.array
            }
        }
        """
        data_dict = dict()
        for file_index, contents_file_path in enumerate(self.contents_file_paths):
            print('[INFO] num of processed novel count: {}'.format(file_index))
            ncode = self.ncode_from_contents_file_path(contents_file_path)
            lines, X_per_novel, Y_per_novel = self.create_non_seq_tensors_emb_cossim_per_novel(ncode=ncode)
            if Y_per_novel is None:
                continue
            per_novel_dict = {'lines': lines, 'X': X_per_novel, 'Y': Y_per_novel}
            data_dict[ncode] = per_novel_dict

            # 100作品ごとにdictを保存する
            if file_index % 10 == 0:
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
        {
        ncode:
            {
            lines: np.array,
            X: np.array,
            Y: np.array
            }
        }
        """
        is_tensor_data_exist = os.path.isfile(self.non_seq_data_dict_emb_cossim_path)
        if is_tensor_data_exist and not tensor_refresh:
            data_dict = self.load_non_seq_data_dict_emb_cossim_data()
        else:
            data_dict = self.create_non_seq_data_dict_emb_cossim()
        return data_dict

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    X: 文中の単語ベクトルの平均ベクトル
    Y: 本文中の文が各あらすじ文と最も類似しているかをOneHotVectorで表現
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""

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
        contents_lines = self.get_contents_lines(ncode=ncode)
        synopsis_lines = self.get_synopsis_lines(ncode=ncode)
        if not synopsis_lines or not contents_lines:
            return None, None
        contents_BoW_vectors, synopsis_BoW_vectors = self.get_BoW_vectors(contents_lines=contents_lines, synopsis_lines=synopsis_lines)
        error_line_indexes = []
        for line_idx, contents_line in enumerate(contents_lines):
            if line_idx % 30 == 0:
                print('{} progress X: {:.1f}%'.format(ncode, line_idx / len(contents_lines) * 100))
            # 本文各文の文ベクトルをX_per_novelに追加
            try:
                sentence_vector = self.get_avg_word_vectors(contents_line)
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
            similarity = [self.cos_sim(contents_BoW_vector, synopsis_BoW_vector) for contents_BoW_vector in contents_BoW_vectors]
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
        for file_index, contents_file_path in enumerate(self.contents_file_paths):
            print('[INFO] num of processed novel count: {}'.format(file_index))
            ncode = self.ncode_from_contents_file_path(contents_file_path)
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

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""
    訓練データの加工
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def dict_train_test_split(self, data_dict, splited_refresh=False, test_size=0.2):
        """
        TrainとTestに分割されたデータを返す
        :param data_dict: dict
        分割する辞書データ
        :param splited_refresh: Bool
        分割する際のNコードをリフレッシュするか
        :param test_size: float
        テストデータの割合
        :return: (dict, dict)
        """
        print('spliting data dict train test...')
        is_ncodes_file_exists = os.path.isfile(self.non_seq_data_dict_emb_cossim_train_ncode_path) \
                                and os.path.isfile(self.non_seq_data_dict_emb_cossim_test_ncode_path)
        if is_ncodes_file_exists and not splited_refresh:
            with open(self.non_seq_data_dict_emb_cossim_train_ncode_path, 'rb') as train_f:
                train_ncodes = joblib.load(train_f)
            with open(self.non_seq_data_dict_emb_cossim_test_ncode_path, 'rb') as test_f:
                test_ncodes = joblib.load(test_f)
        else:
            train_ncodes = list(data_dict.keys())[:int(len(data_dict) * (1 - test_size))]
            test_ncodes = list(data_dict.keys())[int(len(data_dict) * (1 - test_size)):]
            print('saving splited data ncode...')
            with open(self.non_seq_data_dict_emb_cossim_train_ncode_path, 'wb') as train_f:
                joblib.dump(train_ncodes, train_f, compress=3)
            with open(self.non_seq_data_dict_emb_cossim_test_ncode_path, 'wb') as test_f:
                joblib.dump(test_ncodes, test_f, compress=3)
        train_data = {ncode: data_dict[ncode] for ncode in train_ncodes}
        test_data = {ncode: data_dict[ncode] for ncode in test_ncodes}
        return train_data, test_data

    def data_dict_to_tensor(self, data_dict):
        """
        データの辞書をまとめてテンソルに変換
        :param data_dict: dict
        {
        ncode:
            {
            X: np.array,
            Y: np.array
            }
        }
        :return: (np.array, np.array)
        """
        X_values = [value['X'] for value in data_dict.values()]
        Y_values = [value['Y'] for value in data_dict.values()]
        X_tensor = np.array(list(chain.from_iterable(X_values)))
        Y_tensor = np.array(list(chain.from_iterable(Y_values)))
        return X_tensor, Y_tensor

if __name__ == '__main__':
    corpus = NarouCorpus()
    corpus.non_seq_data_dict_emb_cossim(tensor_refresh=True)
    # corpus.non_seq_data_dict_emb_one_of_k(tensor_refresh=True)
    # corpus.create_non_seq_tensors_emb_one_of_k_per_novel(ncode='n6921cw')