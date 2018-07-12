import os
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from src.util import settings
from src.narou.corpus.narou_corpus import NarouCorpus

class KerasExtractiveSummarizer:

    def __init__(self):

        # NAROU CORPUS
        corpus = NarouCorpus()

        # TRAINING DATA
        self.X, self.Y = corpus.non_seq_tensor_emb_cossim()

        # DNN MODEL PROPERTY
        self.n_in = corpus.sentence_vector_size
        self.n_hiddens = [200, 200]
        self.n_out = 1
        self.activation = 'relu'
        self.p_keep = 0.5
        np.random.seed(123)

        # MULTI REGRESSION MODEL
        # self.model = self.multi_reg_model()

        # TRAINED ESTIMATOR
        self.estimator = None



    def multi_reg_model(self):
        """
        DNNで重回帰分析を行うモデルを構築する
        :return: Sequential
        """
        def weight_variable(shape, name=None):
            return np.sqrt(2.0 / shape[0]) * np.random.normal(size=shape)

        model = Sequential()
        for i, input_dim in enumerate(([self.n_in] + self.n_hiddens)[:-1]):
            model.add(Dense(self.n_hiddens[i], input_dim=input_dim,
                            kernel_initializer=weight_variable))
            model.add(BatchNormalization())
            model.add(Activation(self.activation))
            model.add(Dropout(self.p_keep))
        model.add(Dense(self.n_out))
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])
        return model

    def fit(self):
        """
        Feed Foward Neural Netを用いた訓練
        :return:
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        epochs = 50
        batch_size = 10000
        model = self.multi_reg_model()
        X_train, X_test, Y_train, Y_test = \
            train_test_split(self.X, self.Y, test_size=0.2)
        X_train, X_validation, Y_train, Y_validation = \
            train_test_split(X_train, Y_train, test_size=0.4)
        hist = model.fit(X_train, Y_train, epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(X_validation, Y_validation),
                         callbacks=[early_stopping])
        return hist

    def fit_multi_reg(self):
        """
        KerasRegressorを用いた重回帰分析の訓練を行う
        """
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.1, random_state=0)
        estimator = KerasRegressor(build_fn=self.multi_reg_model, epochs=30, batch_size=10000, verbose=0)
        estimator.fit(X_train, Y_train)
        self.estimator = estimator
        self.predict(X_test=X_test, Y_test=Y_test)

    def predict(self, X_test, Y_test):
        """
        学習したモデルを用いて推定を行い平均二乘誤差の平方根を出力
        """
        if not self.estimator:
            print("haven't trained yet")
            return
        Y_pred = self.estimator.predict(X_test)
        print(Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)
        print("KERAS REG RMSE : %.4f" % (mse ** 0.5))

if __name__ == '__main__':
    summarizer = KerasExtractiveSummarizer()
    hist = summarizer.fit()
    acc = hist.history['val_acc']
    loss = hist.history['val_loss']

    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.plot(range(len(loss)), loss,
             label='loss', color='black')
    plt.xlabel('epochs')
    plt.show()



