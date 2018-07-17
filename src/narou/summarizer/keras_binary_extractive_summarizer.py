import os
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

from src.util import settings
from src.narou.corpus.narou_corpus import NarouCorpus


class KerasBinaryExtractiveSummarizer:

    def __init__(self):

        # PATH
        # self.trained_model_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'bin_trained_model', '.hdf5')
        self.trained_model_path = None
        # NAROU CORPUS
        self.corpus = NarouCorpus()

        # TRAINING DATA
        self.data_dict = self.corpus.non_seq_data_dict_emb_one_of_k(tensor_refresh=False)
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

    def inference(self):
        """
        DNNで２値分類を行うモデルを構築する
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
        Feed Foward Neural Netを用いた訓練
        :return:
        """
        print('training data count: {}'.format(len(self.X_train)))
        epochs = 1000
        batch_size = 100
        model = self.inference()

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=10,
                                       verbose=1)
        checkpoint = ModelCheckpoint(filepath=os.path.join(settings.NAROU_MODEL_DIR_PATH,
                                                           'bin_trained_model',
                                                           'model_{epoch:02d}_vloss{val_loss:.4f}.hdf5'),
                                     save_best_only=True)
        hist = model.fit(self.X_train, self.Y_train, epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(self.X_validation, self.Y_validation),
                         callbacks=[early_stopping, checkpoint])
        self.trained_model = model
        self.training_hist = hist



