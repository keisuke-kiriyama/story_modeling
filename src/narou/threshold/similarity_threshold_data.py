import os
import numpy as np
import joblib

from src.util import settings
from src.narou.corpus.embedding_and_bin_classified_sentence_data import EmbeddingAndBinClassifiedSentenceData

class SimilarityThresholdData:
    """
    小説本文の各文に付与されたスコアとそのスコアで採用するか否かの閾値のデータを返す
    X: スコアの分布を表すベクトルを要素とするarray
    Y: 閾値を要素とするarray
    """

    def __init__(self):
        # PATH
        self.similarity_threshold_X_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'similarity_threshold_data', 'similarity_threshold_X.txt')
        self.similarity_threshold_Y_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'similarity_threshold_data', 'similarity_threshold_Y.txt')

        # DATA SUPPLIER
        self.data_supplier = EmbeddingAndBinClassifiedSentenceData()

        # PROPERTY
        self.input_vector_size = 103

    def load_data(self):
        print('loading similarity threshold data...')
        with open(self.similarity_threshold_X_path, 'rb') as Xf:
            X = joblib.load(Xf)
        with open(self.similarity_threshold_Y_path, 'rb')  as Yf:
            Y = joblib.load(Yf)
        return X, Y

    def create_vector_per_novel(self, data):
        """
        一つの小説のベクトルを作成する
        X: 本文各文に付与されたスコアのヒストグラム + avg + std
        Y: 閾値
        :return:
        """
        scores = data['Y_score']
        avg = np.average(scores)
        std = np.std(scores)
        max = np.max(scores)
        min = np.min(scores)
        # 0~1で交差0.01の等差数列を境界としてscoresのヒストグラムを作る
        bins = np.arange(0, 1, 0.01)
        hist, _ = np.histogram(scores, bins=bins, density=True)
        hist = hist / 100
        X = np.append(hist, avg)
        X = np.append(X, std)
        X = np.append(X, max)
        X = np.append(X, min)
        Y = data['threshold']
        return X, Y

    def create_data(self):
        base_data_dict = self.data_supplier.embedding_and_bin_classified_sentence_data_dict(data_refresh=False)
        X = np.empty((0, self.input_vector_size), float)
        Y = np.array([])
        for index, (ncode, data) in enumerate(base_data_dict.items()):
            print('[INFO] num of processed novel count: {}'.format(index))
            X_per_nover, Y_per_novel = self.create_vector_per_novel(data)
            X = np.append(X, [X_per_nover], axis=0)
            Y = np.append(Y, Y_per_novel)
        print("saving similarity threshold data...")
        with open(self.similarity_threshold_X_path, 'wb') as Xf:
            joblib.dump(X, Xf, compress=3)
        with open(self.similarity_threshold_Y_path, 'wb') as Yf:
            joblib.dump(Y, Yf, compress=3)
        return X, Y


    def similarity_threshold_data(self, data_refresh=False):
        """
        データを新たに作るか、ロードするか判別し、データを返す
        """
        is_data_exist = os.path.isfile(self.similarity_threshold_X_path)\
                        and os.path.isfile(self.similarity_threshold_Y_path)
        if is_data_exist and not data_refresh:
            X, Y = self.load_data()
        else:
            X, Y = self.create_data()
        return X, Y

if __name__ == '__main__':
    thresh = SimilarityThresholdData()
    thresh.create_data()

