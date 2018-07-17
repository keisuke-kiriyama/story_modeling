import os
import json
from itertools import chain
import numpy as np
import re
import MeCab
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import corpora, matutils
from rouge import Rouge
from src.util import settings
from src.narou.corpus.narou_corpus import NarouCorpus

class SynopsisSentenceVerificator:

    def __init__(self):
        self.corpus = NarouCorpus()
        self.model_output_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'doc2vec.model')

    def create_doc_embedding_model(self):
        """
        Doc2Vecで文の分散表現を取得するためのモデルの構築
        """
        labeled_sentences = []
        for i, contents_file_path in enumerate(self.corpus.contents_file_paths):
            print('loading data progress: {}'.format(i/len(self.corpus.contents_file_paths)))
            ncode = self.corpus.ncode_from_contents_file_path(contents_file_path)
            wakati_contents_lines = self.corpus.get_wakati_contents_lines(ncode=ncode)
            contents_labeled_sentences = [LabeledSentence(words, tags=[ncode + '_' + str(i)]) for i, words in
                                          enumerate(wakati_contents_lines)]
            wakati_synopsis_lines = self.corpus.get_wakati_synopsis_lines(ncode)
            synopsis_labeled_sentences = [LabeledSentence(words, tags=[ncode + '_synopsis_' + str(i)]) for i, words in
                                          enumerate(wakati_synopsis_lines)]
            labeled_sentences.extend(contents_labeled_sentences + synopsis_labeled_sentences)
        print('training...')
        model = Doc2Vec(dm=0, vector_size=300, window=5, alpha=.025, min_alpha=.025, min_count=1, workers=4, epoch=1)
        model.build_vocab(labeled_sentences)
        epoch_count = 20
        for epoch in range(epoch_count):
            print('Epoch: {0} / {1}'.format(epoch + 1, epoch_count))
            model.train(labeled_sentences, total_examples=model.corpus_count,epochs=model.epochs)
            model.alpha -= (0.025 - 0.0001) / (epoch_count - 1)
            model.min_alpha = model.alpha
        model.save(self.model_output_path)

    def verificate_synopsis_vector_similarity(self, ncode):
        """
        Doc2Vecによる文の分散表現で本文とあらすじ文の類似度を検証する
        :param ncode: str
        検証するNcode
        """
        wakati_contents_lines = self.corpus.get_wakati_contents_lines(ncode=ncode)
        wakati_synopsis_lines = self.corpus.get_wakati_synopsis_lines(ncode=ncode)
        model = Doc2Vec.load(self.model_output_path)
        show_simirality_rank = 3
        for synopsis_idx, synopsis_line in enumerate(wakati_synopsis_lines):
            similarity_dict = {}
            for contents_idx, contents_line in enumerate(wakati_contents_lines):
                similarity_dict[contents_idx] = model.docvecs.similarity_unseen_docs(model,
                                                                                     synopsis_line,
                                                                                     contents_line,
                                                                                     alpha = 1,
                                                                                     min_alpha=0.0001,
                                                                                     steps=5)
            print('-' * 60)
            print('synopsis index: {}'.format(synopsis_idx))
            print(''.join(wakati_synopsis_lines[synopsis_idx]))
            for rank in range(show_simirality_rank):
                sentence_idx, simirality = max(similarity_dict.items(), key=lambda x: x[1])
                print('similarity: {}'.format(simirality))
                print(''.join(wakati_contents_lines[sentence_idx]))
                similarity_dict.pop(sentence_idx)
            print('\n')
        return model

    def verificate_synopsis_BoW_simirality(self, ncode):
        """
        文のBag-of-Wordsベクトルで本文とあらすじ文の類似度を検証する
        :param ncode: str
        検証するNcode
        """
        contents_lines = self.corpus.get_contents_lines(ncode=ncode)
        synopsis_lines = self.corpus.get_synopsis_lines(ncode=ncode)
        contents_vectors, synopsis_vectors = self.corpus.get_BoW_vectors(contents_lines, synopsis_lines)
        show_simirality_rank = 1
        for synopsis_idx, synopsis_vector in enumerate(synopsis_vectors):
            print('-' * 60)
            print(self.corpus.cleaning(''.join(synopsis_lines[synopsis_idx])) + '\n')
            sim = {}
            for contens_idx, contents_vector in enumerate(contents_vectors):
                sim[contens_idx] = self.corpus.cos_sim(synopsis_vector, contents_vector)
            for rank in range(show_simirality_rank):
                sentence_idx, simirality = max(sim.items(), key=lambda x: x[1])
                print('similarity: {:.3f}'.format(simirality))
                print(''.join(contents_lines[sentence_idx]))
                sim.pop(sentence_idx)
            print('\n')

    def get_contents_lines_max_similarity(self, contents_lines, synopsis_lines):
        """
        本文各文に最大類似度を付与して返却
        :param ncode: str
        :return: (np.array, np.array)
        """
        contents_vectors, synopsis_vectors = self.corpus.get_BoW_vectors(contents_lines, synopsis_lines)
        contents_lines_max_simirality = np.array([])
        for contents_vector in contents_vectors:
            max_sim = 0
            for synopsis_vector in synopsis_vectors:
                sim = self.corpus.cos_sim(contents_vector, synopsis_vector)
                if sim > max_sim:
                    max_sim = sim
            contents_lines_max_simirality = np.append(contents_lines_max_simirality, max_sim)
        return contents_lines_max_simirality


    def verificate_more_most_similar_sentences(self, ncode, sentence_count):
        """
        本文の各文にあらすじ文との類似度の最大値をラベルづけし、
        上位から数件取得することであらすじらしいものが取得できるかを検証
        :param ncode: str
        :param sentence_count: int
        生成するあらすじの文数
        """
        contents_lines = self.corpus.get_contents_lines(ncode=ncode)
        synopsis_lines = self.corpus.get_synopsis_lines(ncode=ncode)
        contents_lines_max_similarity = self.get_contents_lines_max_similarity(contents_lines=contents_lines,
                                                                              synopsis_lines=synopsis_lines)

        similar_sentence_indexes = np.argpartition(-contents_lines_max_similarity,
                                                   sentence_count)[:sentence_count]
        appear_ordered = np.sort(similar_sentence_indexes)
        for sentence_index in appear_ordered:
            print(contents_lines[sentence_index])
            print(contents_lines_max_similarity[sentence_index])

    def verificate_all_sentences_similarity(self, ncode):
        contents_lines = self.corpus.get_contents_lines(ncode=ncode)
        synopsis_lines = self.corpus.get_synopsis_lines(ncode=ncode)
        contents_lines_max_similarity = self.get_contents_lines_max_similarity(contents_lines=contents_lines,
                                                                              synopsis_lines=synopsis_lines)

        # 全文の類似度を確認
        for contents_line, similarity in zip(contents_lines, contents_lines_max_similarity):
            print('- ' + contents_line)
            print(similarity)

    def verificate_similar_synopsis_rouge(self, ncode, sentence_count):
        contents_lines = self.corpus.get_contents_lines(ncode=ncode)
        synopsis_lines = self.corpus.get_synopsis_lines(ncode=ncode)
        contents_lines_max_similarity = self.get_contents_lines_max_similarity(contents_lines=contents_lines,
                                                                              synopsis_lines=synopsis_lines)
        similar_sentence_indexes = np.argpartition(-contents_lines_max_similarity,
                                                   sentence_count)[:sentence_count]
        appear_ordered = np.sort(similar_sentence_indexes)
        similar_synopsis = ''.join([contents_lines[index].replace('\u3000', '') for index in appear_ordered])
        correct_synopsis = ''.join(synopsis_lines).replace('\n', '。')
        self.show_rouge_score(syn=similar_synopsis, ref=correct_synopsis)

    def show_rouge_score(self, syn, ref):
        # 正解のあらすじと類似度により作成されたあらすじ間のRougeスコアを求める
        rouge = Rouge()
        wakati_similar_synopsis = self.corpus.wakati(syn)
        wakati_correct_synopsis = self.corpus.wakati(ref)
        score = rouge.get_scores(wakati_similar_synopsis, wakati_correct_synopsis)
        print('[CORRECT SYNOPSIS]: {}'.format(ref))
        print('[SIMILAR SYNOPSIS]: {}'.format(syn))
        print('[ROUGE-1]')
        print('f-measure: {}'.format(score[0]['rouge-1']['f']))
        print('precision: {}'.format(score[0]['rouge-1']['r']))
        print('recall: {}'.format(score[0]['rouge-1']['p']))
        print('[ROUGE-2]')
        print('f-measure: {}'.format(score[0]['rouge-2']['f']))
        print('precision: {}'.format(score[0]['rouge-2']['r']))
        print('recall: {}'.format(score[0]['rouge-2']['p']))
        print('[ROUGE-L]')
        print('f-measure: {}'.format(score[0]['rouge-l']['f']))
        print('precision: {}'.format(score[0]['rouge-l']['r']))
        print('recall: {}'.format(score[0]['rouge-l']['p']))


if __name__ == '__main__':
    verificator = SynopsisSentenceVerificator()
    # verificator.create_doc_embedding_model()
    # verificator.verificate_synopsis_vector_similarity('n0002ei')
    # verificator.verificate_synopsis_BoW_simirality('n9974br')
    verificator.verificate_similar_synopsis_rouge('n9974br', 2)
