import os
import sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from rouge import Rouge
import joblib
from itertools import chain

from src.util import settings
from src.narou.corpus.embedding_and_bin_classified_sentence_data import EmbeddingAndBinClassifiedSentenceData

class KerasMultiSentencesAdoptedSummarizer:

    def __init__(self):
        # PATH
        self.trained_model_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'msa_trained_model', '180809','')
        self.train_data_ncodes_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'msa_trained_model', 'train_data_ncodes.txt')
        self.test_data_ncodes_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'msa_trained_model', 'test_data_ncodes.txt')

        # DATA SUPPLIER
        self.data_supplier = EmbeddingAndBinClassifiedSentenceData()

        # CORPUS PROCESSOR
        self.corpus = self.data_supplier.corpus

        # DATA
        raw_data_dict = self.data_supplier.embedding_and_bin_classified_sentence_data_dict(data_refresh=False)
        self.data_dict = self.data_screening(raw_data_dict, rouge_lower_limit=0.35)
        self.train_data_ncodes, self.test_data_ncodes = self.ncodes_train_test_split()
        self.X_train, self.Y_train = self.data_dict_to_tensor(ncodes=self.train_data_ncodes)
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(self.X_train, self.Y_train, test_size=0.2)
        self.X_test, self.Y_test = self.data_dict_to_tensor(ncodes=self.test_data_ncodes)

        # DNN MODEL PROPERTY
        self.n_in = self.data_supplier.sentence_vector_size
        self.n_hiddens = [800, 800]
        self.n_out = 2
        self.activation = 'relu'
        self.p_keep = 0.5

        # TRAINED MODEL
        if os.path.isfile(self.trained_model_path):
            print('loading trained model: {}'.format(self.trained_model_path))
            self.trained_model = load_model(self.trained_model_path)
        else:
            self.trained_model = None
        self.training_hist = None


    def data_screening(self, raw_data_dict, rouge_lower_limit):
        """
        あらすじに小説の本文と似た表現を用いていない小説を除外する
        付与されたOpt Rougeの値で足切りする
        :param raw_data_dict: dict
        :param rouge_lower_limit: float
        :return: dict
        """
        data_dict = raw_data_dict.copy()
        for ncode in raw_data_dict.keys():
            if raw_data_dict[ncode]['rouge']['f'] < rouge_lower_limit:
                del data_dict[ncode]
        return data_dict

    def ncodes_train_test_split(self, ncode_refresh=False, test_size=0.2):
        """
        訓練データとテストデータにそれぞれに対応するNコードを記録
        """
        is_ncodes_file_exists = os.path.isfile(self.train_data_ncodes_path) \
                                and os.path.isfile(self.test_data_ncodes_path)
        if is_ncodes_file_exists and not ncode_refresh:
            print('loading splited data ncode...')
            with open(self.train_data_ncodes_path, 'rb') as train_f:
                train_ncodes = joblib.load(train_f)
            with open(self.test_data_ncodes_path, 'rb') as test_f:
                test_ncodes = joblib.load(test_f)
        else:
            ncodes = list(self.data_dict.keys())
            train_ncodes = ncodes[:int(len(ncodes) * (1 - test_size))]
            test_ncodes = ncodes[int(len(ncodes) * (1 - test_size)):]
            print('saving splited data ncode...')
            with open(self.train_data_ncodes_path, 'wb') as train_f:
                joblib.dump(train_ncodes, train_f, compress=3)
            with open(self.test_data_ncodes_path, 'wb') as test_f:
                joblib.dump(test_ncodes, test_f, compress=3)
        return train_ncodes, test_ncodes

    def data_dict_to_tensor(self, ncodes):
        """
        dictのデータを学習用テンソルに変換する
        :param ncodes: [str]
        :return:
        """
        print('converting dict to tensor...')
        X_tensors = [self.data_dict[ncode]['X'] for ncode in ncodes]
        Y_tensors = [self.data_dict[ncode]['Y'] for ncode in ncodes]
        X_tensor = np.array(list(chain.from_iterable(X_tensors)))
        Y_tensor = np.array(list(chain.from_iterable(Y_tensors)))
        Y_one_of_k_tensor = np.eye(2)[Y_tensor.astype(int)]
        return X_tensor, Y_one_of_k_tensor

    def inference(self):
        """
        DNNで2値分類を行うモデルを構築する
        :return: Sequential
        """
        model = Sequential()
        for i, input_dim in enumerate(([self.n_in] + self.n_hiddens)[:-1]):
            model.add(Dense(self.n_hiddens[i], input_dim=input_dim))
            model.add(BatchNormalization())
            model.add(Activation(self.activation))
            model.add(Dropout(self.p_keep))
        model.add(Dense(self.n_out))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999))
        return model

    def fit(self):
        """
        FFNN fit
        """
        print('training data count: {}'.format(len(self.X_train)))
        epochs = 1000
        batch_size=100
        model = self.inference()

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=10)
        checkpoint = ModelCheckpoint(filepath=os.path.join(settings.NAROU_MODEL_DIR_PATH,
                                                           'msa_trained_model',
                                                           '180809',
                                                           'model_{epoch:02d}_vloss{val_loss:.4f}.hdf5'),
                                     save_best_only=True)
        hist = model.fit(self.X_train, self.Y_train, epochs=epochs,
                         batch_size=batch_size,
                         verbose=1,
                         validation_data=(self.X_validation, self.Y_validation),
                         callbacks=[early_stopping, checkpoint])
        self.trained_model = model
        self.training_hist = hist

if __name__ == '__main__':
    summarizer = KerasMultiSentencesAdoptedSummarizer()
    summarizer.fit()