import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils.generic_utils import get_custom_objects
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from src.util import settings
from src.narou.corpus.narou_corpus import NarouCorpus

class KerasExtractiveSummarizer:

    def __init__(self):
        # PATH
        self.trained_model_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'keras_extractive_summarizer_model.hdf5')

        # NAROU CORPUS
        self.corpus = NarouCorpus()

        # TRAINING DATA
        self.data_dict = self.corpus.non_seq_data_dict_emb_cossim(tensor_refresh=False)
        self.training_data_dict, self.test_data_dict = self.corpus.dict_train_test_split(self.data_dict, test_size=0.2)
        self.X_train, self.Y_train = self.corpus.data_dict_to_tensor(data_dict=self.training_data_dict)
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(self.X_train, self.Y_train, test_size=0.2)
        self.X_test, self.Y_test = self.corpus.data_dict_to_tensor(data_dict=self.test_data_dict)

        # DNN MODEL PROPERTY
        self.n_in = self.corpus.sentence_vector_size
        self.n_hiddens = [800, 800]
        self.n_out = 1
        self.activation = 'relu'
        self.p_keep = 0.5

        # TRAINED
        if os.path.isfile(self.trained_model_path):
            self.trained_model = load_model(self.trained_model_path)
        else:
            self.trained_model = None
        self.training_hist = None

    def weight_variable(self, shape):
            return np.sqrt(2.0 / shape[0]) * np.random.normal(size=shape)

    def inference(self):
        """
        DNNで重回帰分析を行うモデルを構築する
        :return: Sequential
        """
        model = Sequential()
        for i, input_dim in enumerate(([self.n_in] + self.n_hiddens)[:-1]):
            model.add(Dense(self.n_hiddens[i], input_dim=input_dim))
            model.add(BatchNormalization())
            model.add(Activation(self.activation))
            model.add(Dropout(self.p_keep))
        model.add(Dense(self.n_out))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999))
        return model

    def fit(self):
        """
        Feed Foward Neural Netを用いた訓練
        :return:
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        epochs = 1000
        batch_size = 100
        model = self.inference()

        hist = model.fit(self.X_train, self.Y_train, epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(self.X_validation, self.Y_validation),
                         callbacks=[early_stopping])
        model.save(self.trained_model_path)
        self.trained_model = model
        self.training_hist = hist

    def eval(self):
        """
        学習済みモデルの評価
        """
        if not self.trained_model:
            print("haven't trained yet")
            return
        Y_pred = self.trained_model.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, Y_pred)
        print("MEAN SQUARED ERROR : %.4f" % (mse ** 0.5))

    def generate_synopsis(self, ncode, sentence_count, sim_threshold):
        if ncode in self.training_data_dict.keys():
            print('Exist in training data set')
            X = self.training_data_dict[ncode]['X']
            Y = self.training_data_dict[ncode]['X']
        elif ncode in self.test_data_dict.keys():
            print('Exist in test data set')
            X = self.test_data_dict[ncode]['X']
            Y = self.test_data_dict[ncode]['Y']
        else:
            print('Not exist in data set')
            X, Y = self.corpus.create_non_seq_tensors_emb_cossim_per_novel(ncode=ncode)
        Y_pred = self.trained_model.predict(X)
        similar_sentence_indexes = np.argpartition(-Y_pred.T,
                                                   sentence_count)[0][:sentence_count]
        contents_lines = self.corpus.get_contents_lines(ncode=ncode)
        synopsis_lines = [self.corpus.cleaning(contents_lines[line_index]) for line_index in similar_sentence_indexes]
        synopsis = ''.join(synopsis_lines)
        return synopsis

    def show_training_process(self):
        """
        訓練過程の損失関数の値をプロット
        """
        loss = summarizer.training_hist.history['val_loss']

        plt.rc('font', family='serif')
        plt.plot(range(len(loss)), loss,
                 label='loss', color='black')
        plt.xlabel('epochs')
        plt.show()

    def verificate_synopsis_generation(self):
        """
        テスト用に作成されたテンソルを用いて実際に出力されるあらすじを確認する
        """
        test_ncodes = self.test_data_dict.keys()
        synopsis_sentence_count = 8
        for test_ncode in test_ncodes:
            print('[INFO] test ncode: {}'.format(test_ncode))
            contents_lines = self.corpus.get_contents_lines(ncode=test_ncode, is_test_data=False)
            X = self.test_data_dict[test_ncode]['X']
            Y = self.test_data_dict[test_ncode]['Y']
            Y_pred = self.trained_model.predict(X)
            mse = mean_squared_error(Y, Y_pred)

            # 本文全文に付与された値と正解のcos類似度を出力
            print('mean squared error = {}'.format(mse))
            for i, pred in enumerate(Y_pred):
                print(contents_lines[i])
                print('prediction: {:.3f}'.format(float(pred)))
                print('correct similarity: {:.3f}'.format(Y[i]))
                print('\n')

            # 付与された値が高い順に文を出力
            # すなわち実際に生成されるあらすじを出力する
            print('-' * 100)
            similar_sentence_indexes = np.argpartition(-Y_pred.T,
                                                   synopsis_sentence_count)[0][:synopsis_sentence_count]
            appear_ordered = np.sort(similar_sentence_indexes)
            for sentence_index in appear_ordered:
                print(contents_lines[sentence_index])
                print('similarity: {}'.format(Y_pred[sentence_index][0]))
                print('correct siilarity: {}'.format(Y[sentence_index]))
                print('\n')

if __name__ == '__main__':
    summarizer = KerasExtractiveSummarizer()
    # summarizer.fit()
    # summarizer.eval()
    # summarizer.show_training_process()
    # summarizer.verificate_synopsis_generation()
    # summarizer.generate_synopsis('n0011cx', sentence_count=8, sim_threshold=0.3)
