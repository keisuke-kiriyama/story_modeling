import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from src.narou.corpus.narou_corpus import NarouCorpus



def train_emb_idx():
    corpus = NarouCorpus()
    N = len(corpus.contents_file_paths)
    N_train = int(N * 0.9)
    # N_validation = N - N_train
    contents_length = 1000
    synopsis_length = 300
    X, Y = corpus.data_to_tensor_emb_idx(contents_length=contents_length, synopsis_length=synopsis_length)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, train_size = N_train)

    # モデル設定
    n_in = corpus.embedding_size
    n_hidden = 128
    n_out = 1

    model = Sequential()

    # Encoder

    model.add(LSTM(n_hidden, input_shape=(contents_length, n_in)))

    # Decoder

    model.add(RepeatVector(synopsis_length))
    model.add(LSTM(n_hidden, return_sequences=True))

    model.add(TimeDistributed(Dense(n_out)))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])

    # モデル学習
    epochs = 30
    batch_size = 5

    hist = model.fit(X_train, Y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(X_validation, Y_validation))
    return hist



if __name__ == '__main__':
    hist = train_emb_idx()
    acc = hist.history['val_acc']
    loss = hist.history['val_loss']

    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.plot(range(len(loss)), loss,
             label='loss', color='black')
    plt.xlabel('epochs')
    plt.show()
