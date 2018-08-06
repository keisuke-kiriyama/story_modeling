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

from src.util import settings
from src.narou.corpus.narou_corpus import NarouCorpus

class KerasRegressionExtractiveSummarizer:

    def __init__(self):
        # PATH
        self.trained_model_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'reg_trained_model', '180727','model_04_vloss0.0060.hdf5')

        # NAROU CORPUS
        self.corpus = NarouCorpus()

        # TRAINING DATA
        self.data_dict = self.corpus.non_seq_data_dict_emb_cossim(tensor_refresh=False)
        self.training_data_dict, self.test_data_dict = self.corpus.dict_train_test_split(self.data_dict, splited_refresh=False, test_size=0.2)
        self.X_train, self.Y_train = self.corpus.data_dict_to_tensor(data_dict=self.training_data_dict)
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(self.X_train, self.Y_train, test_size=0.2)
        self.X_test, self.Y_test = self.corpus.data_dict_to_tensor(data_dict=self.test_data_dict)
        print('num of novels which is training data: {}'.format(len(self.training_data_dict.keys())))
        print('num of novels which is test data: {}'.format(len(self.test_data_dict.keys())))

        # DNN MODEL PROPERTY
        self.n_in = self.corpus.sentence_vector_size
        self.n_hiddens = [800, 800]
        self.n_out = 1
        self.activation = 'relu'
        self.p_keep = 0.5

        # TRAINED
        if os.path.isfile(self.trained_model_path):
            print('loading trained model: {}'.format(self.trained_model_path))
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
                                       patience=10)
        checkpoint = ModelCheckpoint(filepath=os.path.join(settings.NAROU_MODEL_DIR_PATH,
                                                           'reg_trained_model',
                                                           '180727',
                                                           'model_{epoch:02d}_vloss{val_loss:.4f}.hdf5'),
                                     save_best_only=True)
        hist = model.fit(self.X_train, self.Y_train, epochs=epochs,
                         batch_size=batch_size,
                         verbose=1,
                         validation_data=(self.X_validation, self.Y_validation),
                         callbacks=[early_stopping, checkpoint])
        self.trained_model = model
        self.training_hist = hist

    def predict_synopsis(self, ncode, sentence_count):
        """
        ncodeの小説のあらすじを学習済みモデルから生成する
        :param ncode: str
        :param sentence_count: int
        :return: str
        """
        if ncode in self.training_data_dict.keys():
            X = self.training_data_dict[ncode]['X']
            error_line_indexes = self.training_data_dict[ncode]['error_line_indexes']
        elif ncode in self.test_data_dict.keys():
            X = self.test_data_dict[ncode]['X']
            error_line_indexes = self.training_data_dict[ncode]['error_line_indexes']
        else:
            print('nothing ncode in data_dict')
            return
        Y_pred = self.trained_model.predict(X)
        similar_sentence_indexes = np.argpartition(-Y_pred.T,
                                                   sentence_count)[0][:sentence_count]
        contents_lines = self.corpus.get_contents_lines(ncode=ncode)
        removed_contents_lines = self.corpus.remove_error_line_indexes_from_contents_lines(contents_lines=contents_lines,
                                                                                           error_line_indexes=error_line_indexes)
        synopsis_lines = [self.corpus.cleaning(removed_contents_lines[line_index]) for line_index in similar_sentence_indexes]
        synopsis = ''.join(synopsis_lines)
        return synopsis

    def predict_synopsis_with_threshold(self, ncode, threshold):
        """
        閾値以上の類似度が推測された分からあらすじを生成する
        """
        if ncode in self.training_data_dict.keys():
            X = self.training_data_dict[ncode]['X']
            error_line_indexes = self.training_data_dict[ncode]['error_line_indexes']
        elif ncode in self.test_data_dict.keys():
            X = self.test_data_dict[ncode]['X']
            error_line_indexes = self.training_data_dict[ncode]['error_line_indexes']
        else:
            print('nothing ncode in data_dict')
            return
        Y_pred = self.trained_model.predict(X)
        contents_lines = np.array(self.corpus.get_contents_lines(ncode=ncode))
        removed_contents_lines = self.corpus.remove_error_line_indexes_from_contents_lines(contents_lines=contents_lines,
                                                                                           error_line_indexes=error_line_indexes)
        synopsis_lines = removed_contents_lines[np.where(Y_pred>0.2)[0]]
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
        opts = [] # 類似度上位から文選択(理論上の上限値)
        leads = [] # 文章の先頭から数文選択
        pros = [] # 提案手法要約

        for ncode, test_data in self.test_data_dict.items():
            # 正解と同じ文数取得
            ref, opt, lead, pro = self.create_synopsises_same_count_test_data(ncode=ncode, test_data=test_data)
            # 一定数文取得
            # ref, opt, lead, pro = self.create_synopsis_fixed_count(ncode=ncode, test_data=test_data, sentence_count=5)
            # 閾値以上の文取得
            # ref, opt, lead, pro = self.create_synopsis_above_similarity_threshold(ncode=ncode, test_data=test_data, threshold=0.4)

            refs.append(ref)
            opts.append(opt)
            leads.append(lead)
            pros.append(pro)

        sys.setrecursionlimit(20000)
        rouge = Rouge()
        # OPTIMAL EVALUATION
        scores = rouge.get_scores(opts, refs, avg=True)
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
        print('\n')

        # LEAD EVALUATION
        scores = rouge.get_scores(leads, refs, avg=True)
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
        print('\n')

        # PROPOSED EVALUATION
        scores = rouge.get_scores(pros, refs, avg=True)
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
        # print('\n')

    def create_synopsises_same_count_test_data(self, ncode, test_data):
        """
        正解のあらすじ文と同じ文それぞれの分かち書きされた各あらすじを返す
        """
        # 正解のあらすじと同じ文数
        # refs
        correct_synopsis_lines = self.corpus.get_synopsis_lines(ncode=ncode)
        correct_synopsis = ''.join(correct_synopsis_lines)
        ref = self.corpus.wakati(correct_synopsis)
        # opt
        contents_lines = self.corpus.get_contents_lines(ncode=ncode)
        removed_contents_lines = self.corpus.remove_error_line_indexes_from_contents_lines(contents_lines,
                                                                                           test_data['error_line_indexes'])
        similar_sentence_indexes = np.argpartition(-test_data['Y'],
                                               len(correct_synopsis_lines))[:len(correct_synopsis_lines)]
        appear_ordered = np.sort(similar_sentence_indexes)
        opt_lines = [removed_contents_lines[index] for index in appear_ordered]
        opt_synopsis = ''.join(opt_lines)
        opt = self.corpus.wakati(opt_synopsis)
        # lead
        # lead_synopsis = ''.join([self.corpus.cleaning(line) for line in contents_lines[:len(correct_synopsis_lines)]])
        # lead = self.corpus.wakati(lead_synopsis)
        # proposed
        # predict_synopsis = self.predict_synopsis(ncode=ncode, sentence_count=len(correct_synopsis_lines))
        # pro = self.corpus.wakati(predict_synopsis)
        lead = 'test'
        pro = 'test'
        return ref, opt, lead, pro

    def create_synopsis_fixed_count(self, ncode, test_data, sentence_count):
        """
        一定文数によって作成された各あらすじを返す
        """
        # refs
        correct_synopsis_lines = self.corpus.get_synopsis_lines(ncode=ncode)
        correct_synopsis = ''.join(correct_synopsis_lines)
        ref = self.corpus.wakati(correct_synopsis)

        # opt
        contents_lines = self.corpus.get_contents_lines(ncode=ncode)
        similar_sentence_indexes = np.argpartition(-test_data['Y'],
                                               sentence_count)[:sentence_count]
        appear_ordered = np.sort(similar_sentence_indexes)
        opt_lines = [contents_lines[index] for index in appear_ordered]
        opt_synopsis = ''.join(opt_lines)
        opt = self.corpus.wakati(opt_synopsis)

        # lead
        lead_synopsis = ''.join([self.corpus.cleaning(line) for line in contents_lines[:sentence_count]])
        lead = self.corpus.wakati(lead_synopsis)

        # proposed
        predict_synopsis = self.predict_synopsis(ncode=ncode, sentence_count=sentence_count)
        pro = self.corpus.wakati(predict_synopsis)
        return ref, opt, lead, pro

    def create_synopsis_above_similarity_threshold(self, ncode, test_data, threshold):
        """
        閾値以上の値が付与された文を選択し、各あらすじを返す
        """
        # refs
        correct_synopsis_lines = self.corpus.get_synopsis_lines(ncode=ncode)
        correct_synopsis = ''.join(correct_synopsis_lines)
        ref = self.corpus.wakati(correct_synopsis)

        # opt
        contents_lines = np.array(self.corpus.get_contents_lines(ncode=ncode))
        opt_lines = contents_lines[np.where(test_data['Y'] > threshold)]
        opt_synopsis = ''.join(opt_lines)
        opt = self.corpus.wakati(opt_synopsis)

        # lead
        lead_synopsis = ''.join([self.corpus.cleaning(line) for line in contents_lines[:len(opt_lines)]])
        lead = self.corpus.wakati(lead_synopsis)

        # proposed
        predict_synopsis = self.predict_synopsis_with_threshold(ncode=ncode, threshold=threshold)
        pro = self.corpus.wakati(predict_synopsis)
        return ref, opt, lead, pro


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
            contents_lines = self.corpus.get_contents_lines(ncode=test_ncode)
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
    summarizer = KerasRegressionExtractiveSummarizer()
    # summarizer.fit()
    summarizer.eval()
    # summarizer.show_training_process()
    # summarizer.verificate_synopsis_generation()
    # summarizer.generate_synopsis('n0011cx', sentence_count=8, sim_threshold=0.3)
