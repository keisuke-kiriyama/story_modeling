import os
import sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_recall_curve, auc
import matplotlib.pyplot as plt
from rouge import Rouge
import joblib
from itertools import chain
import click

from src.util import settings
from src.narou.corpus.multi_feature_and_bin_classified_sentence_data import MultiFeatureAndBinClassifiedSentenceData
from src.narou.threshold.threshold_estimator import ThresholdEstimator

@click.group()
def cmd():
    pass

class KerasRegressionExtractiveSummarizer:

    def __init__(self, genre, trained_model_path):
        # PATH
        self.trained_model_path = trained_model_path
        self.train_data_ncodes_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'ncodes', genre, 'train_data_ncodes.txt')
        self.test_data_ncodes_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'ncodes', genre, 'test_data_ncodes.txt')

        # DATA SUPPLIER
        self.data_supplier = MultiFeatureAndBinClassifiedSentenceData()

        # CORPUS PROCESSOR
        self.corpus = self.data_supplier.corpus

        # THRESHOLD ESTIMATOR
        self.threshold_estimator = ThresholdEstimator()

        # DATA
        raw_data_dict = self.data_supplier.multi_feature_and_bin_classified_sentence_data_dict(data_refresh=False)
        self.data_dict = self.data_screening(raw_data_dict, rouge_lower_limit=0.35)
        self.train_data_ncodes, self.test_data_ncodes = self.ncodes_train_test_split(ncode_refresh=False)
        self.X_train, self.Y_train = self.data_dict_to_tensor(ncodes=self.train_data_ncodes)
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(self.X_train, self.Y_train, test_size=0.2)
        self.X_test, self.Y_test = self.data_dict_to_tensor(ncodes=self.test_data_ncodes)
        print('num of novels which is training data: {}'.format(len(self.train_data_ncodes)))
        print('num of novels which is test data: {}'.format(len(self.test_data_ncodes)))

        # DNN MODEL PROPERTY
        self.n_in = self.data_supplier.input_vector_size
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

        # PROPERTY
        self.genre = genre

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
        Y_tensors = [self.data_dict[ncode]['Y_score'] for ncode in ncodes]
        X_tensor = np.array(list(chain.from_iterable(X_tensors)))
        Y_tensor = np.array(list(chain.from_iterable(Y_tensors)))
        return X_tensor, Y_tensor

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
                                                           self.genre,
                                                           '180904',
                                                           'model_{epoch:02d}_vloss{val_loss:.4f}.hdf5'),
                                     save_best_only=True)
        hist = model.fit(self.X_train, self.Y_train, epochs=epochs,
                         batch_size=batch_size,
                         verbose=1,
                         validation_data=(self.X_validation, self.Y_validation),
                         callbacks=[early_stopping, checkpoint])
        self.trained_model = model
        self.training_hist = hist

    def eval_pr_curve(self):
        """
        precision-recall曲線のAUCを用いてスコア付のモデルの評価を行う
        """
        pro_aucs = []
        lead_aucs = []
        for ncode in self.test_data_ncodes:
            # 提案モデルのAUC
            data = self.data_dict[ncode]
            y_true = data['Y']
            pro_y_scores = np.array(list(chain.from_iterable(self.trained_model.predict(data['X']))))
            pro_precision, pro_recall, _ = precision_recall_curve(y_true, pro_y_scores)
            pro_auc = auc(pro_recall, pro_precision)
            pro_aucs.append(pro_auc)

            # 先頭から順に文を採用した場合のAUC
            lead_y_scores = np.arange(len(data['Y']))
            lead_precision, lead_recall, _ = precision_recall_curve(y_true, lead_y_scores)
            lead_auc = auc(lead_recall, lead_precision)
            lead_aucs.append(lead_auc)
        print('[PRO AUC] {}'.format(np.mean(pro_aucs)))
        print('[LEAD AUC] {}'.format(np.mean(lead_aucs)))

    def eval_rouge(self):
        """
        学習済みモデルの評価
        - 平均２乘誤差
        - ROUGE-1
        - ROUGE-2
        - ROUGE-L
        """
        print('evaluating...')
        if not self.trained_model:
            print("haven't trained yet")
            return
        Y_pred = self.trained_model.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, Y_pred)
        print("[INFO] MEAN SQUARED ERROR : %.4f" % (mse ** 0.5))

        # ROUGE
        refs = []               # 参照要約
        opts = []               # 類似度上位から文選択(理論上の上限値)
        leads = []              # 文章の先頭からoptの文数分選択
        pros = []               # 提案手法要約
        fives = []              # 回帰の結果から上位5文取得
        same_opt_counts = []    # optのあらすじ文数と同じ文数を採用して生成したあらすじ

        # COMPRESSION RATIO
        comp_ratios = []

        for ncode in self.test_data_ncodes:
            ref = self.generate_ref_synopsis(ncode)
            opt = self.generate_opt_synopsis(ncode)
            lead = self.generate_lead_synopsis(ncode)
            pro = self.generate_pro_synopsis(ncode)
            counts = self.generate_counts_synopsis(ncode, max_sentence_count=6)
            if pro == '':
                pro = '*'

            refs.append(ref)
            opts.append(opt)
            leads.append(lead)
            pros.append(pro)
            fives.append(counts[4])
            same_opt_counts.append(counts[-1])

            wakati_len = len(counts[-1].replace(' ', ''))
            contents_len = len(''.join(self.corpus.get_contents_lines(ncode)))
            cr = wakati_len / contents_len
            comp_ratios.append(cr)
        print('compression ratio: {}'.format(np.average(comp_ratios)))

        sys.setrecursionlimit(20000)
        rouge = Rouge()
        # OPTIMAL EVALUATION
        scores = rouge.get_scores(opts, refs, avg=True)
        print('[OPTIMAL]')
        print('[ROUGE-1]')
        print('f-measure: {:.3f}'.format(scores['rouge-1']['f']))
        print('precision: {:.3f}'.format(scores['rouge-1']['r']))
        print('recall: {:.3f}'.format(scores['rouge-1']['p']))
        print('[ROUGE-2]')
        print('f-measure: {:.3f}'.format(scores['rouge-2']['f']))
        print('precision: {:.3f}'.format(scores['rouge-2']['r']))
        print('recall: {:.3f}'.format(scores['rouge-2']['p']))
        print('\n')

        # LEAD EVALUATION
        scores = rouge.get_scores(leads, refs, avg=True)
        print('[LEAD]')
        print('[ROUGE-1]')
        print('f-measure: {:.3f}'.format(scores['rouge-1']['f']))
        print('precision: {:.3f}'.format(scores['rouge-1']['r']))
        print('recall: {:.3f}'.format(scores['rouge-1']['p']))
        print('[ROUGE-2]')
        print('f-measure: {:.3f}'.format(scores['rouge-2']['f']))
        print('precision: {:.3f}'.format(scores['rouge-2']['r']))
        print('recall: {:.3f}'.format(scores['rouge-2']['p']))
        print('\n')

        # PROPOSED EVALUATION
        scores = rouge.get_scores(pros, refs, avg=True)
        print('[PROPOSED METHOD EVALUATION]')
        print('[ROUGE-1]')
        print('f-measure: {:.3f}'.format(scores['rouge-1']['f']))
        print('precision: {:.3f}'.format(scores['rouge-1']['r']))
        print('recall: {:.3f}'.format(scores['rouge-1']['p']))
        print('[ROUGE-2]')
        print('f-measure: {:.3f}'.format(scores['rouge-2']['f']))
        print('precision: {:.3f}'.format(scores['rouge-2']['r']))
        print('recall: {:.3f}'.format(scores['rouge-2']['p']))
        print('\n')

        # SAME OPT COUNTS
        scores = rouge.get_scores(same_opt_counts, refs, avg=True)
        print('[SAME OPT SENTENCE ADOPTED EVALUATION]')
        print('[ROUGE-1]')
        print('f-measure: {:.3f}'.format(scores['rouge-1']['f']))
        print('precision: {:.3f}'.format(scores['rouge-1']['r']))
        print('recall: {:.3f}'.format(scores['rouge-1']['p']))
        print('[ROUGE-2]')
        print('f-measure: {:.3f}'.format(scores['rouge-2']['f']))
        print('precision: {:.3f}'.format(scores['rouge-2']['r']))
        print('recall: {:.3f}'.format(scores['rouge-2']['p']))
        print('\n')


    def generate_ref_synopsis(self, ncode):
        """
        参照要約を生成
        """
        synopsis_lines = self.corpus.get_synopsis_lines(ncode)
        wakati_synopsis = self.corpus.wakati(''.join(synopsis_lines))
        return wakati_synopsis

    def generate_opt_synopsis(self, ncode):
        """
        本文から重要文を選択する際の最も理想的な選択によりあらすじを生成
        """
        data = self.data_dict[ncode]
        contents_lines = self.corpus.get_contents_lines(ncode)
        removed_contents_lines = np.array(self.corpus.remove_error_line_indexes_from_contents_lines(contents_lines,
                                                                                           data['error_line_indexes']))
        positive_sentence_index = np.where(data['Y']==1)
        opt_synopsis = self.corpus.wakati(''.join(removed_contents_lines[positive_sentence_index]))
        return opt_synopsis

    def generate_lead_synopsis(self, ncode):
        """
        本文先頭から正解のあらすじ文と同じ文数選択することによりあらすじを生成
        """
        # sentence_count = len(self.corpus.get_synopsis_lines(ncode))
        sentence_count = len(np.where(self.data_dict[ncode]['Y'] == 1)[0])
        contents_lines = self.corpus.get_contents_lines(ncode)
        lead_synopsis_lines = contents_lines[:sentence_count]
        lead_synopsis = self.corpus.wakati(''.join(lead_synopsis_lines))
        return lead_synopsis

    def generate_pro_synopsis(self, ncode):
        """
        提案手法の学習済みモデルによりあらすじの生成を行う
        """
        data = self.data_dict[ncode]
        Y_pred = self.trained_model.predict(data['X'])
        Y_pred = np.array(list(chain.from_iterable(Y_pred)))

        # 閾値推定モデルの入力の形に変換し閾値を推定する
        threshold_estimator_input = self.convert_to_threshold_estimator_input(scores=Y_pred)
        threshold_estimator_input = np.reshape(threshold_estimator_input, (1, self.threshold_estimator.n_in))
        threshold = self.threshold_estimator.trained_model.predict(threshold_estimator_input)[0]

        contents_lines = self.corpus.get_contents_lines(ncode)
        removed_contents_lines = np.array(self.corpus.remove_error_line_indexes_from_contents_lines(contents_lines,
                                                                                           error_line_indexes=data['error_line_indexes']))
        pro_synopsis_lines = removed_contents_lines[np.where(Y_pred > threshold)]
        pro_synopsis = self.corpus.wakati(''.join(pro_synopsis_lines))
        return pro_synopsis

    def generate_counts_synopsis(self, ncode, max_sentence_count=5):
        """
        回帰結果から上位数文選択してあらすじのリストを返す
        :param max_sentence_count: int
        :return: list
        """
        data = self.data_dict[ncode]
        Y_pred = self.trained_model.predict(data['X'])
        Y_pred = np.array(list(chain.from_iterable(Y_pred)))

        contents_lines = self.corpus.get_contents_lines(ncode)
        removed_contents_lines = np.array(self.corpus.remove_error_line_indexes_from_contents_lines(contents_lines,
                                                                                           error_line_indexes=data['error_line_indexes']))

        higher_score_indexes = np.argsort(-Y_pred)[:max_sentence_count]
        higher_score_slices = [np.array(higher_score_indexes[:i+1]) for i in range(len(higher_score_indexes))]
        synopsises_lines = [removed_contents_lines[indices] for indices in higher_score_slices]
        synopsises = [self.corpus.wakati(''.join(synopsis)) for synopsis in synopsises_lines]

        # 正解のあらすじ文数と同じ文数のあらすじ生成
        sentence_count = len(np.where(self.data_dict[ncode]['Y'] == 1)[0])
        higher_score_slice = np.array(higher_score_indexes[:sentence_count])
        synopsis_line = removed_contents_lines[higher_score_slice]
        synopsises = np.append(synopsises, self.corpus.wakati(''.join(synopsis_line)))

        return synopsises

    def show_training_process(self):
        """
        訓練過程の損失関数の値をプロット
        """
        loss = self.training_hist.history['val_loss']

        plt.rc('font', family='serif')
        plt.plot(range(len(loss)), loss,
                 label='loss', color='black')
        plt.xlabel('epochs')
        plt.show()

    def convert_to_threshold_estimator_input(self, scores):
        """
        類似度のベクトルを閾値推定モデルのインプットのベクトルに変換する
        :param scores:  np.array
        :return: np.array
        """
        avg = np.average(scores)
        std = np.std(scores)
        max = np.max(scores)
        min = np.min(scores)
        # 0~1で交差0.01の等差数列を境界としてscoresのヒストグラムを作る
        bins = np.arange(0, 1, 0.01)
        hist, _ = np.histogram(scores, bins=bins, density=True)
        hist = hist / 100
        input = np.append(hist, avg)
        input = np.append(input, std)
        input = np.append(input, max)
        input = np.append(input, min)
        return input

@cmd.command()
@click.option('--genre', '-g', default='general')
def full(genre):
    print('[INFO] Genre is {}'.format(genre))
    trained_model_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'reg_trained_model', genre, '180817','model_26_vloss0.0058.hdf5')
    summarizer = KerasRegressionExtractiveSummarizer(genre, trained_model_path)
    summarizer.fit()
    summarizer.eval_rouge()
    summarizer.eval_pr_curve()

@cmd.command()
@click.option('--genre', '-g', default='general')
def eval_only(genre):
    print('[INFO] Genre is {}'.format(genre))
    trained_model_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'reg_trained_model', genre, '180822','model_02_vloss0.0064.hdf5')
    summarizer = KerasRegressionExtractiveSummarizer(genre, trained_model_path)
    summarizer.eval_rouge()
    # summarizer.eval_pr_curve()

def main():
    cmd()

if __name__ == '__main__':
    main()