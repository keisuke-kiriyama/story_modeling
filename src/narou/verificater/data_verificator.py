import os

import numpy as np
from rouge import Rouge

from src.util import settings
from src.narou.corpus.narou_corpus import NarouCorpus

class DataVerificator:

    def __init__(self):
        self.corpus = NarouCorpus()
        self.data_dict = self.corpus.non_seq_data_dict_emb_cossim(tensor_refresh=False)
        self.init_scores()

    def init_scores(self):
        """
        スコアの分布をまとめるリストをまとめる
        """
        self.vh_f, self.h_f, self.m_f, self.l_f, self.vl_f = [], [], [], [], []
        self.vh_r, self.h_r, self.m_r, self.l_r, self.vl_r = [], [], [], [], []
        self.vh_p, self.h_p, self.m_p, self.l_p, self.vl_p = [], [], [], [], []
        self.error_ncodes = []

    def score_assignment(self, ncode, f_score, recall, precision):
        """
        スコアを分配する
        :param ncode: str
        :param f_score: float
        :param recall: float
        :param precision: float
        """
        if f_score >= 0.5:
            self.vh_f.append(ncode)
        elif 0.5 > f_score and f_score >= 0.4:
            self.h_f.append(ncode)
        elif 0.4 > f_score and f_score >= 0.3:
            self.m_f.append(ncode)
        elif 0.3 > f_score and f_score >= 0.2:
            self.l_f.append(ncode)
        elif 0.2 > f_score:
            self.vl_f.append(ncode)
        if recall >= 0.5:
            self.vh_r.append(ncode)
        elif 0.5 > recall and recall >= 0.4:
            self.h_r.append(ncode)
        elif 0.4 > recall and recall >= 0.3:
            self.m_r.append(ncode)
        elif 0.3 > recall and recall >= 0.2:
            self.l_r.append(ncode)
        elif 0.2 > recall:
            self.vl_r.append(ncode)
        if precision >= 0.5:
            self.vh_p.append(ncode)
        elif 0.5 > precision and precision >= 0.4:
            self.h_p.append(ncode)
        elif 0.4 > precision and precision >= 0.3:
            self.m_p.append(ncode)
        elif 0.3 > precision and precision >= 0.2:
            self.l_p.append(ncode)
        elif 0.2 > precision:
            self.vl_p.append(ncode)

    def verificate_opt_synopsis_generation_same_sentences_count_as_ref_per_novel(self, ncode):
        rouge = Rouge()
        synopsis_lines = self.corpus.get_synopsis_lines(ncode=ncode)
        if not synopsis_lines:
            self.error_ncodes.append(ncode)
            return
        contents_lines = self.corpus.get_contents_lines(ncode=ncode)
        if not contents_lines:
            self.error_ncodes.append(ncode)
            return
        if len(synopsis_lines) > len(contents_lines):
            self.error_ncodes.append(ncode)
            return
        ref_synopsis = ''.join(synopsis_lines)
        wakati_ref_synopsis = self.corpus.wakati(ref_synopsis)
        sentences_count = len(synopsis_lines)
        data = self.data_dict[ncode]
        similar_sentence_indexes = np.argpartition(-data['Y'],
                                                   sentences_count)[:sentences_count]


        removed_contents_lines = self.corpus.remove_error_line_indexes_from_contents_lines(contents_lines,
                                                                                           data['error_line_indexes'])
        generated_synopsis = ''.join([removed_contents_lines[line_index] for line_index in similar_sentence_indexes])
        wakati_generated_synopsis = self.corpus.wakati(generated_synopsis)
        try:
            score = rouge.get_scores(wakati_generated_synopsis, wakati_ref_synopsis)
        except RecursionError as err:
            self.error_ncodes.append(ncode)
            return
        f_score = score[0]['rouge-1']['f']
        recall = score[0]['rouge-1']['r']
        precision = score[0]['rouge-1']['r']
        self.score_assignment(ncode=ncode, f_score=f_score, recall=recall, precision=precision)


    def verificate_opt_synopsis_generation_same_sentences_count_as_ref(self):
        """
        参照のあらすじ文数と同じ文数で付与された重要度スコアからあらすじを生成し、スコアの分布を調べる
        """
        self.init_scores()
        for i, ncode in enumerate(self.data_dict.keys()):
            print('{} progress: {:.1f}%'.format(ncode, i / len(self.data_dict) * 100))
            self.verificate_opt_synopsis_generation_same_sentences_count_as_ref_per_novel(ncode=ncode)


    def verificate_opt_synopsis_generation_by_threshold_per_novel(self, data, wakati_ref_synopsis, threshold=0.5):
        """
        閾値以上の重要度スコアからあらすじを生成しスコアの分布を調べる
        :param threshold: float
        """
        rouge = Rouge()
        ncode = data['ncode']
        contents_lines = self.corpus.get_contents_lines(ncode=ncode)
        removed_contents_lines = self.corpus.remove_error_line_indexes_from_contents_lines(contents_lines,
                                                                                           data['error_line_indexes'])
        generated_synopsis = ''.join(removed_contents_lines[np.where(data['Y'] > threshold)[0]])
        wakati_generated_synopsis = self.corpus.wakati(generated_synopsis)
        score = rouge.get_scores(wakati_generated_synopsis, wakati_ref_synopsis)
        f_score = score['rouge-1']['f']
        recall = score['rouge-1']['r']
        precision = score['rouge-1']['r']

    def verificate_opt_synopsis_generation_by_threshold(self):
        for i, data in enumerate(self.data_dict):
            ncode = data['ncode']
            synopsis_lines = self.corpus.get_synopsis_lines(ncode=ncode)
            ref_synopsis = ''.join(synopsis_lines)
            wakati_ref_synopsis = self.corpus.wakati(ref_synopsis)

            contents_lines = self.corpus.remove_error_line_indexes_from_contents_lines(contents_lines=contents_lines,
                                                                                       error_line_indexes=data['error_line_indexes'])


    def show_f_score_dist(self):
        print("very high f-score: {}".format(len(self.vh_f)))
        print(self.vh_f)
        print('\n')
        print("high f-score: {}".format(len(self.h_f)))
        print(self.h_f)
        print('\n')
        print("medium f-score: {}".format(len(self.m_f)))
        print(self.m_f)
        print('\n')
        print("low f-score: {}".format(len(self.l_f)))
        print(self.l_f)
        print('\n')
        print("very low f-score: {}".format(len(self.vl_f)))
        print(self.vl_f)
        print('\n')

    def show_recall_dist(self):
        print("very high recall: {}".format(len(self.vh_r)))
        print(self.vh_r)
        print('\n')
        print("high recall: {}".format(len(self.h_r)))
        print(self.h_r)
        print('\n')
        print("medium recall: {}".format(len(self.m_r)))
        print(self.m_r)
        print('\n')
        print("low recall: {}".format(len(self.l_r)))
        print(self.l_r)
        print('\n')
        print("very low recall: {}".format(len(self.vl_r)))
        print(self.vl_r)
        print('\n')

    def show_precision_dist(self):
        print("very high precision: {}".format(len(self.vh_p)))
        print(self.vh_p)
        print('\n')
        print("high precision: {}".format(len(self.h_p)))
        print(self.h_p)
        print('\n')
        print("medium precision: {}".format(len(self.m_p)))
        print(self.m_p)
        print('\n')
        print("low precision: {}".format(len(self.l_p)))
        print(self.l_p)
        print('\n')
        print("very low precision: {}".format(len(self.vl_p)))
        print(self.vl_p)
        print('\n')



if __name__ == '__main__':
    verificator = DataVerificator()

