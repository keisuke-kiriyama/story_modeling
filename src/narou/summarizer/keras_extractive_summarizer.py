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
        corpus = NarouCorpus()

        # TRAINING DATA
        self.X, self.Y = corpus.non_seq_tensor_emb_cossim()
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(self.X, self.Y, test_size=0.1)
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = \
            train_test_split(self.X_train, self.Y_train, test_size=0.1)

        # DNN MODEL PROPERTY
        self.n_in = corpus.sentence_vector_size
        self.n_hiddens = [200, 200]
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
        epochs = 20
        batch_size = 10000
        model = self.inference()

        hist = model.fit(self.X_train, self.Y_train, epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(self.X_validation, self.Y_validation),
                         callbacks=[early_stopping])
        model.save(self.trained_model_path)
        self.trained_model = model
        self.training_hist = hist

    def predict(self):
        """
        学習したモデルを用いて推定を行い平均二乘誤差の平方根を出力
        """
        if not self.trained_model:
            print("haven't trained yet")
            return
        Y_pred = self.trained_model.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, Y_pred)
        print("KERAS REG RMSE : %.4f" % (mse ** 0.5))

    def show_training_process(self):
        """
        訓練過程の損失関数の値をプロット
        :return:
        """
        loss = summarizer.training_hist.history['val_loss']

        plt.rc('font', family='serif')
        plt.plot(range(len(loss)), loss,
                 label='loss', color='black')
        plt.xlabel('epochs')
        plt.show()

if __name__ == '__main__':
    summarizer = KerasExtractiveSummarizer()
    summarizer.fit()
    summarizer.predict()
    # summarizer.show_training_process()



