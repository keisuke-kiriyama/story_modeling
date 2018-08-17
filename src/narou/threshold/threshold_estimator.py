import os
import numpy as np
import joblib
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.util import settings
from src.narou.threshold.similarity_threshold_data import SimilarityThresholdData

class ThresholdEstimator:
    """
    閾値を推定するモデル
    """

    def __init__(self):
        # PATH
        self.trained_model_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'threshold_estimate_trained_model', '180816', 'model_38_vloss0.0057.hdf5')

        # DATA SUPPLIER
        self.data_supplier = SimilarityThresholdData()
        X, Y = self.data_supplier.similarity_threshold_data(data_refresh=False)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.1)
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(self.X_train, self.Y_train, test_size=0.1)

        # DNN MODEL PROPERTY
        self.n_in = self.data_supplier.input_vector_size
        self.n_hiddens = [300, 300]
        self.n_out = 1
        self.activation = 'relu'
        self.p_keep = 0.5

        # TRAINED MODEL
        if os.path.isfile(self.trained_model_path):
            print('loading trained model: {}'.format(self.trained_model_path))
            self.trained_model = load_model(self.trained_model_path)
        else:
            self.trained_model = None
        self.training_hist = None

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
        """
        print('training data count: {}'.format(len(self.X_train)))
        epochs = 1000
        batch_size = 100
        model = self.inference()

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=10)
        checkpoint = ModelCheckpoint(filepath=os.path.join(settings.NAROU_MODEL_DIR_PATH,
                                                           'threshold_estimate_trained_model',
                                                           '180816',
                                                           'model_{epoch:02d}_vloss{val_loss:.4f}.hdf5'),
                                     save_best_only=True)
        hist = model.fit(self.X_train, self.Y_train, epochs=epochs,
                         batch_size=batch_size,
                         verbose=1,
                         validation_data=(self.X_validation, self.Y_validation),
                         callbacks=[early_stopping, checkpoint])
        self.trained_model = model
        self.training_hist = hist

    def eval(self):
        """
        学習済みモデルの評価
        """
        print('evaluating...')
        if not self.trained_model:
            print("haven't trained yet")
            return
        Y_pred = self.trained_model.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, Y_pred)
        print("[INFO] MEAN SQUARED ERROR : %.4f" % (mse ** 0.5))

if __name__ == '__main__':
    estimator = ThresholdEstimator()
    estimator.fit()
    estimator.eval()