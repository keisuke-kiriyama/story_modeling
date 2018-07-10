import os
import json
from itertools import chain
import numpy as np
import re
import MeCab
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import corpora, matutils
from src.util import settings


class SynopsisSentenceVerificator:

    def __init__(self):
        self.novel_contents_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'contents')
        self.novel_meta_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'meta')
        self.contents_file_paths = [os.path.join(self.novel_contents_dir_path, file_name) for file_name in os.listdir(self.novel_contents_dir_path) if not file_name == '.DS_Store']
        self.meta_file_paths = [os.path.join(self.novel_meta_dir_path, file_name) for file_name in os.listdir(self.novel_meta_dir_path) if not file_name == '.DS_Store']
        self.model_output_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'doc2vec.model')


    def ncode_from_contents_file_path(self, file_path):
        return file_path.split('/')[-1].split('.')[0]

    def load(self, file_path):
        json_file = open(file_path, 'r')
        contents = json.load(json_file)
        json_file.close()
        return contents

    def wakati(self, line):
        m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati')
        wakati = m.parse(line)
        return wakati

    def cos_sim(self, v1, v2):
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def create_contents_file_path(self, ncode):
        return os.path.join(self.novel_contents_dir_path, ncode + '.json')

    def create_meta_file_path(self, ncode):
        return os.path.join(self.novel_meta_dir_path, ncode+'_meta.json')

    def cleaning(self, line):
        line = line.replace('\u3000', '')
        line = line.replace('\n', '')
        return line

    def remove_stop_word(self, sentence):
        """
        文中の名詞、形容詞、動詞、副詞のリスト
        :param sentence: str
        :return: list
        """
        part = ['名詞', '動詞', '形容詞', '副詞']
        m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        morphs = m.parse(sentence).split('\n')
        removed = []
        for morph in morphs:
            splited = re.split('[,\t]', morph)
            if len(splited) < 2: continue
            if splited[1] in part:
                removed.append(splited[0])
        return removed

    def get_contents_lines(self, ncode):
        """
        本文全文のリストを返却
        :param contents_file_path: str
        :return: list
        """
        contents_file_path = self.create_contents_file_path(ncode=ncode)
        if not contents_file_path in self.contents_file_paths:
            print('nothing ncode')
            return
        contents_lines = [line for line in list(chain.from_iterable(self.load(contents_file_path)['contents'])) if not line == '']
        return contents_lines

    def get_synopsis_lines(self, ncode):
        """
        あらすじの文のリストを返却
        :param synopsis_file_path: str
        :return: list
        """
        meta_file_path = self.create_meta_file_path(ncode=ncode)
        if not meta_file_path in self.meta_file_paths:
            print('nothing ncode')
            return
        synopsis = self.load(meta_file_path)['story']
        synopsis_lines = [line for line in re.split('[。？]', synopsis) if not line == '']
        return synopsis_lines

    def get_wakati_lines(self, lines):
        """
        文のリストを、分かち書きとクリーニングが行われた文のリストに変換
        :param lines: list
        :return: list
        """
        return [self.cleaning(self.wakati(line)).split() for line in lines]

    def get_wakati_contents_lines(self, ncode):
        """
        本文の各文を分かち書きしたリストを取得
        :param ncode: str
        :return: list
        """
        contents_lines = self.get_contents_lines(ncode=ncode)
        wakati_contents_lines = self.get_wakati_lines(contents_lines)
        return wakati_contents_lines

    def get_wakati_synopsis_lines(self, ncode):
        """
        あらすじの各分を分かち書きしたリストを取得
        :param ncode:
        :return:
        """
        synopsis_lines = self.get_synopsis_lines(ncode=ncode)
        wakati_synopsis_lines = self.get_wakati_lines(synopsis_lines)
        return wakati_synopsis_lines

    def get_BoW_vectors(self, contents_lines, synopsis_lines):
        """
        文のリストから各文のBoWベクトルのリストを返す
        :param contents_lines: list
        本文の各文を要素とするリスト
        :param synopsis_lines: list
        あらすじの各文を要素とするリスト
        :return: ([np.array], [np.array])
        """
        removed_contents_lines = [self.remove_stop_word(self.cleaning(line)) for line in contents_lines]
        removed_synopsis_lines = [self.remove_stop_word(self.cleaning(line)) for line in synopsis_lines]
        all_lines = removed_contents_lines + removed_synopsis_lines
        vocaburaly = corpora.Dictionary(all_lines)
        contents_BoWs = [vocaburaly.doc2bow(line) for line in removed_contents_lines]
        synopsis_BoWs = [vocaburaly.doc2bow(line) for line in removed_synopsis_lines]
        contents_vectors = [np.array(matutils.corpus2dense([bow], num_terms=len(vocaburaly)).T[0]) for bow in contents_BoWs]
        synopsis_vectors = [np.array(matutils.corpus2dense([bow], num_terms=len(vocaburaly)).T[0]) for bow in synopsis_BoWs]
        return contents_vectors, synopsis_vectors

    def create_doc_embedding_model(self):
        """
        Doc2Vecで文の分散表現を取得するためのモデルの構築
        """
        labeled_sentences = []
        for i, contents_file_path in enumerate(self.contents_file_paths):
            print('loading data progress: {}'.format(i/len(self.contents_file_paths)))
            ncode = self.ncode_from_contents_file_path(contents_file_path)
            wakati_contents_lines = self.get_wakati_contents_lines(ncode=ncode)
            contents_labeled_sentences = [LabeledSentence(words, tags=[ncode + '_' + str(i)]) for i, words in
                                          enumerate(wakati_contents_lines)]
            wakati_synopsis_lines = self.get_wakati_synopsis_lines(ncode)
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
        wakati_contents_lines = self.get_wakati_contents_lines(ncode=ncode)
        wakati_synopsis_lines = self.get_wakati_synopsis_lines(ncode=ncode)
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
        contents_lines = self.get_contents_lines(ncode=ncode)
        synopsis_lines = self.get_synopsis_lines(ncode=ncode)
        contents_vectors, synopsis_vectors = self.get_BoW_vectors(contents_lines, synopsis_lines)
        show_simirality_rank = 1
        for synopsis_idx, synopsis_vector in enumerate(synopsis_vectors):
            print('-' * 60)
            print(self.cleaning(''.join(synopsis_lines[synopsis_idx])) + '\n')
            sim = {}
            for contens_idx, contents_vector in enumerate(contents_vectors):
                sim[contens_idx] = self.cos_sim(synopsis_vector, contents_vector)
            for rank in range(show_simirality_rank):
                sentence_idx, simirality = max(sim.items(), key=lambda x: x[1])
                print('similarity: {:.3f}'.format(simirality))
                print(''.join(contents_lines[sentence_idx]))
                sim.pop(sentence_idx)
            print('\n')

    def sim_generate_synopsis_verification(self, ncode, sentence_count):
        """
        本文の各文にあらすじ文との類似度の最大値をラベルづけし、
        上位から数件取得することであらすじらしいものが取得できるかを検証
        :param ncode: str
        :param sentence_count: int
        生成するあらすじの文数
        """
        contents_lines = self.get_contents_lines(ncode=ncode)
        synopsis_lines = self.get_synopsis_lines(ncode=ncode)
        contents_vectors, synopsis_vectors = self.get_BoW_vectors(contents_lines, synopsis_lines)
        contents_line_max_simirality = np.array([])
        for contents_vector in contents_vectors:
            max_sim = 0
            for synopsis_vector in synopsis_vectors:
                sim = self.cos_sim(contents_vector, synopsis_vector)
                if sim > max_sim:
                    max_sim = sim
            contents_line_max_simirality = np.append(contents_line_max_simirality, max_sim)
        similar_sentence_indexes = np.argpartition(-contents_line_max_simirality,
                                                   sentence_count)[:sentence_count]
        appear_ordered = np.sort(similar_sentence_indexes)
        for sentence_index in appear_ordered:
            print(contents_lines[sentence_index])
            print(contents_line_max_simirality[sentence_index])

if __name__ == '__main__':
    verificator = SynopsisSentenceVerificator()
    # verificator.create_doc_embedding_model()
    # verificator.verificate_synopsis_vector_similarity('n0002ei')
    # verificator.verificate_synopsis_BoW_simirality('n9974br')
    verificator.sim_generate_synopsis_verification('n0013da', 8)
