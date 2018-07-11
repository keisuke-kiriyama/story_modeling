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
        self.novel_contents_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'contents_small')
        self.novel_meta_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'meta_small')
        self.contents_file_paths = [os.path.join(self.novel_contents_dir_path, file_name) for file_name in os.listdir(self.novel_contents_dir_path) if not file_name == '.DS_Store']
        self.meta_file_paths = [os.path.join(self.novel_meta_dir_path, file_name) for file_name in os.listdir(self.novel_meta_dir_path) if not file_name == '.DS_Store']
        self.non_seq_tensor_emb_cossim_data_X_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'non_seq_tensors_emb_cossim_X.txt')
        self.non_seq_tensor_emb_cossim_data_Y_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'non_seq_tensors_emb_cossim_Y.txt')

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
        return os.path.join(self.novel_contents_dir_path, ncode + '.json')

    def create_meta_file_path(self, ncode):
        return os.path.join(self.novel_meta_dir_path, ncode+'_meta.json')

    def cleaning(self, line):
        line = line.replace('\u3000', '')
        line = line.replace('\n', '')
        return line

    def cos_sim(self, v1, v2):
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def split_synopsis(self, synopsis):
        """
        あらすじを文ごとに分割する
        :param synopsis: str
        :return: list
        """
        return [line for line in re.split('[。？]', synopsis) if not line == '']

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
        contents_lines = [line for line in list(chain.from_iterable(self.load(contents_file_path)['contents'])) if not self.cleaning(line) == '']
        return contents_lines

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
        synopsis = self.load(meta_file_path)['story']
        synopsis_lines = [self.cleaning(line) for line in self.split_synopsis(synopsis)]
        return synopsis_lines

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

    def load_non_seq_tensors_emb_cossim_data(self):
        print('loading tensor data...')
        with open(self.non_seq_tensor_emb_cossim_data_X_path, 'rb') as Xf:
            X = joblib.load(Xf)
        with open(self.non_seq_tensor_emb_cossim_data_Y_path, 'rb') as Yf:
            Y = joblib.load(Yf)
        return X, Y

    def get_avg_word_vectors(self, sentence):
        """
        文中の各単語の平均ベクトル返却
        :param sentence: str
        :return: np.array
        """
        wakati_sentence = self.wakati(sentence).split()
        word_vectors = np.array([self.word_embedding_model.__dict__['wv'][word] for word in wakati_sentence])
        return np.average(word_vectors, axis=0)

    def create_non_seq_tensors_emb_cossim(self):
        X = np.empty((0, self.sentence_vector_size), float)
        Y = np.array([])
        for file_index, contents_file_path in enumerate(self.contents_file_paths):
            ncode = self.ncode_from_contents_file_path(contents_file_path)
            print('[PROCESS NCODE]: {}'.format(ncode))
            contents_lines = self.get_contents_lines(ncode=ncode)
            synopsis_lines = self.get_synopsis_lines(ncode=ncode)
            contents_BoW_vectors, synopsis_BoW_vectors = self.get_BoW_vectors(contents_lines=contents_lines, synopsis_lines=synopsis_lines)
            for line_idx, (contents_line, contents_BoW_vector) in enumerate(zip(contents_lines, contents_BoW_vectors)):
                if line_idx % 30 == 0:
                    print('{} progress: {:.1f}%'.format(ncode, line_idx / len(contents_lines) * 100))
                # 本文各文の文ベクトルをXに追加
                try:
                    sentence_vector = self.get_avg_word_vectors(contents_line)
                except KeyError as err:
                    print(err)
                    continue
                X = np.append(X, [sentence_vector], axis=0)

                # 各文のあらすじ文との最大cos類似度をYに追加
                max_sim = 0
                for synopsis_BoW_vector in synopsis_BoW_vectors:
                    sim = self.cos_sim(contents_BoW_vector, synopsis_BoW_vector)
                    if sim > max_sim:
                        max_sim = sim
                Y = np.append(Y, max_sim)

            # 100作品ごとにTensorを保存する
            if file_index % 100 == 0:
                print('saving tensor...')
                with open(self.non_seq_tensor_emb_cossim_data_X_path, 'wb') as Xf:
                    joblib.dump(X, Xf, compress=3)
                with open(self.non_seq_tensor_emb_cossim_data_Y_path, 'wb') as Yf:
                    joblib.dump(Y, Yf, compress=3)
        return X, Y


    def non_seq_tensor_emb_cossim(self):
        """
        非系列情報と仮定された文ベクトルとコサイン類似度のTensorを返却
        X. 全小説の全文をベクトルで表した２次元ベクトル
        - 文ベクトルは文中の単語ベクトルの平均ベクトル
        - shape = (小説数*文数, 文ベクトルサイズ)
        Y. 全小説の全文のコサイン類似度を要素とする１次元ベクトル
        :return: (np.array, np.array)
        """
        is_tensor_data_exist = os.path.isfile(self.non_seq_tensor_emb_cossim_data_X_path)\
                               and os.path.isfile(self.non_seq_tensor_emb_cossim_data_Y_path)
        if is_tensor_data_exist:
            X, Y = self.load_non_seq_tensors_emb_cossim_data()
        else:
            X, Y = self.create_non_seq_tensors_emb_cossim()
        return X, Y



if __name__ == '__main__':
    corpus = NarouCorpus()
    corpus.non_seq_tensor_emb_cossim()
