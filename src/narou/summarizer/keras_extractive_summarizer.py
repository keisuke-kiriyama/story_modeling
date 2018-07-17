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

class KerasExtractiveSummarizer:

    def __init__(self):
        # PATH
        self.trained_model_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'trained_model', 'model_54_vloss0.0030.hdf5')

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
        print('training data count: {}'.format(len(self.X_train)))
        epochs = 1000
        batch_size = 100
        model = self.inference()

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=10,
                                       verbose=1)
        checkpoint = ModelCheckpoint(filepath=os.path.join(settings.NAROU_MODEL_DIR_PATH,
                                                           'trained_model',
                                                           'model_{epoch:02d}_vloss{val_loss:.4f}.hdf5'),
                                     save_best_only=True)
        hist = model.fit(self.X_train, self.Y_train, epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(self.X_validation, self.Y_validation),
                         callbacks=[early_stopping, checkpoint])
        self.trained_model = model
        self.training_hist = hist

    def generate_synopsis(self, ncode, sentence_count):
        """
        ncodeの小説のあらすじを学習済みモデルから生成する
        :param ncode: str
        :param sentence_count: int
        :return: str
        """
        if ncode in self.training_data_dict.keys():
            X = self.training_data_dict[ncode]['X']
        elif ncode in self.test_data_dict.keys():
            X = self.test_data_dict[ncode]['X']
        else:
            X, _ = self.corpus.create_non_seq_tensors_emb_cossim_per_novel(ncode=ncode)
        Y_pred = self.trained_model.predict(X)
        similar_sentence_indexes = np.argpartition(-Y_pred.T,
                                                   sentence_count)[0][:sentence_count]
        contents_lines = self.corpus.get_contents_lines(ncode=ncode)
        synopsis_lines = [self.corpus.cleaning(contents_lines[line_index]) for line_index in similar_sentence_indexes]
        synopsis = ''.join(synopsis_lines)
        return synopsis

    def eval(self):
        """
        学習済みモデルの評価
        - 平均２乘誤差
        - ROUGE-1
        - ROUGE-2
        - ROUGE-L
        """
        if not self.trained_model:
            print("haven't trained yet")
            return
        Y_pred = self.trained_model.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, Y_pred)
        print("[INFO] MEAN SQUARED ERROR : %.4f" % (mse ** 0.5))

        # PROPOSED
        refs = [] # 参照要約
        opt = [] # 類似度上位から文選択(理論上の上限値)
        lead = [] # 文章の先頭から数文選択
        hyps = [] # 提案手法要約
        for ncode, test_data in self.test_data_dict.items():
            # refs
            correct_synopsis_lines = self.corpus.get_synopsis_lines(ncode=ncode)
            correct_synopsis = ''.join(correct_synopsis_lines)
            wakati_correct_synopsis = self.corpus.wakati(correct_synopsis)
            refs.append(wakati_correct_synopsis)

            # opt
            contents_lines = self.corpus.get_contents_lines(ncode=ncode)
            similar_sentence_indexes = np.argpartition(-test_data['Y'],
                                                   len(correct_synopsis_lines))[:len(correct_synopsis_lines)]
            appear_ordered = np.sort(similar_sentence_indexes)
            opt_lines = [contents_lines[index] for index in appear_ordered]
            opt_synopsis = ''.join(opt_lines)
            wakati_opt_synopsis = self.corpus.wakati(opt_synopsis)
            opt.append(wakati_opt_synopsis)

            # lead
            lead_synopsis = ''.join([self.corpus.cleaning(line) for line in contents_lines[:len(correct_synopsis_lines)]])
            wakati_lead_synopsis = self.corpus.wakati(lead_synopsis)
            print(wakati_lead_synopsis)
            lead.append(wakati_lead_synopsis)

            # proposed
            predict_synopsis = self.generate_synopsis(ncode=ncode, sentence_count=len(correct_synopsis_lines))
            wakati_predict_synopsis = self.corpus.wakati(predict_synopsis)
            hyps.append(wakati_predict_synopsis)

        rouge = Rouge()
        # OPTIMAL EVALUATION
        scores = rouge.get_scores(opt, refs, avg=True)
        print('[OPTIMAL]')
        print('[ROUGE-1]')
        print('f-measure: {}'.format(scores['rouge-1']['f']))
        print('precision: {}'.format(scores['rouge-1']['r']))
        print('recall: {}'.format(scores['rouge-1']['p']))
        print('[ROUGE-2]')
        print('f-measure: {}'.format(scores['rouge-2']['f']))
        print('precision: {}'.format(scores['rouge-2']['r']))
        print('recall: {}'.format(scores['rouge-2']['p']))
        print('[ROUGE-L]')
        print('f-measure: {}'.format(scores['rouge-l']['f']))
        print('precision: {}'.format(scores['rouge-l']['r']))
        print('recall: {}'.format(scores['rouge-l']['p']))

        # LEAD EVALUATION
        scores = rouge.get_scores(lead, refs, avg=True)
        print('[LEAD]')
        print('[ROUGE-1]')
        print('f-measure: {}'.format(scores['rouge-1']['f']))
        print('precision: {}'.format(scores['rouge-1']['r']))
        print('recall: {}'.format(scores['rouge-1']['p']))
        print('[ROUGE-2]')
        print('f-measure: {}'.format(scores['rouge-2']['f']))
        print('precision: {}'.format(scores['rouge-2']['r']))
        print('recall: {}'.format(scores['rouge-2']['p']))
        print('[ROUGE-L]')
        print('f-measure: {}'.format(scores['rouge-l']['f']))
        print('precision: {}'.format(scores['rouge-l']['r']))
        print('recall: {}'.format(scores['rouge-l']['p']))
        
        # PROPOSED EVALUATION
        scores = rouge.get_scores(hyps, refs, avg=True)
        print('[PROPOSED METHOD EVALUATION]')
        print('[ROUGE-1]')
        print('f-measure: {}'.format(scores['rouge-1']['f']))
        print('precision: {}'.format(scores['rouge-1']['r']))
        print('recall: {}'.format(scores['rouge-1']['p']))
        print('[ROUGE-2]')
        print('f-measure: {}'.format(scores['rouge-2']['f']))
        print('precision: {}'.format(scores['rouge-2']['r']))
        print('recall: {}'.format(scores['rouge-2']['p']))
        print('[ROUGE-L]')
        print('f-measure: {}'.format(scores['rouge-l']['f']))
        print('precision: {}'.format(scores['rouge-l']['r']))
        print('recall: {}'.format(scores['rouge-l']['p']))

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
    summarizer.eval()
    # summarizer.show_training_process()
    # summarizer.verificate_synopsis_generation()
    # summarizer.generate_synopsis('n0011cx', sentence_count=8, sim_threshold=0.3)
