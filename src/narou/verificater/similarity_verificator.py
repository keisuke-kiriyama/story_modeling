import os
import json
from itertools import chain
import numpy as np
import re
import joblib
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
        self.labeled_sentences = None

    def create_doc_embedding_model(self, load_labeled_sentences=True):
        labeled_sentences = []
        for i, contents_file_path in enumerate(self.contents_file_paths):
            print('loading data progress: {}'.format(i/len(self.contents_file_paths)))
            ncode = self.ncode_from_contents_file_path(contents_file_path)
            meta_file_path = os.path.join(self.novel_meta_dir_path, ncode+'_meta.json')
            contents_lines = list(chain.from_iterable(self.load(contents_file_path)['contents']))
            synopsis = self.load(meta_file_path)['story']
            synopsis_lines = re.split('[。？]', synopsis)
            wakati_contents_lines = [self.cleaning(self.wakati(line)).split() for line in contents_lines]
            wakati_synopsis_lines = [self.cleaning(self.wakati(line)).split() for line in synopsis_lines]
            contents_labeled_sentences = [LabeledSentence(words, tags=[ncode + '_' + str(i)]) for i, words in
                                          enumerate(wakati_contents_lines)]
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
        verificate_contents_file_path = os.path.join(self.novel_contents_dir_path, ncode + '.json')
        verificate_meta_file_path = os.path.join(self.novel_meta_dir_path, ncode + '_meta.json')
        if not verificate_contents_file_path in self.contents_file_paths or not verificate_meta_file_path in self.meta_file_paths:
            print('nothing ncode')
            return
        contents_lines = list(chain.from_iterable(self.load(verificate_contents_file_path)['contents']))
        synopsis = self.load(verificate_meta_file_path)['story']
        synopsis_lines = re.split('[。？]', synopsis)
        wakati_contents_lines = [self.cleaning(self.wakati(line)).split() for line in contents_lines]
        wakati_synopsis_lines = [self.cleaning(self.wakati(line)).split() for line in synopsis_lines]

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
        contents_file_path = os.path.join(self.novel_contents_dir_path, ncode + '.json')
        meta_file_path = os.path.join(self.novel_meta_dir_path, ncode + '_meta.json')
        if not contents_file_path in self.contents_file_paths or not meta_file_path in self.meta_file_paths:
            print('nothing ncode')
            return
        contents_lines = list(chain.from_iterable(self.load(contents_file_path)['contents']))
        removed_contents_lines = [self.remove_stop_word(self.cleaning(line)) for line in contents_lines]
        synopsis = self.load(meta_file_path)['story']
        synopsis_lines = re.split('[。？]', synopsis)
        removed_synopsis_lines = [self.remove_stop_word(self.cleaning(line)) for line in synopsis_lines]
        all_lines = removed_contents_lines + removed_synopsis_lines
        vocaburaly = corpora.Dictionary(all_lines)
        contents_BoW = [vocaburaly.doc2bow(line) for line in removed_contents_lines]
        synopsis_BoW = [vocaburaly.doc2bow(line) for line in removed_synopsis_lines]
        contents_vectors = [np.array(matutils.corpus2dense([bow], num_terms=len(vocaburaly)).T[0]) for bow in contents_BoW]
        synopsis_vectors = [np.array(matutils.corpus2dense([bow], num_terms=len(vocaburaly)).T[0]) for bow in synopsis_BoW]
        show_simirality_rank = 1
        for synopsis_idx, synopsis_vector in enumerate(synopsis_vectors):
            print('-' * 60)
            # print('synopsis sentence index: {}'.format(synopsis_idx))
            print(self.cleaning(''.join(synopsis_lines[synopsis_idx])) + '\n')
            # print(removed_synopsis_lines[synopsis_idx])
            sim = {}
            for contens_idx, contents_vector in enumerate(contents_vectors):
                sim[contens_idx] = self.cos_sim(synopsis_vector, contents_vector)
            for rank in range(show_simirality_rank):
                sentence_idx, simirality = max(sim.items(), key=lambda x: x[1])
                print('similarity: {:.3f}'.format(simirality))
                print(''.join(contents_lines[sentence_idx]))
                # print(removed_contents_lines[sentence_idx])
                sim.pop(sentence_idx)
            print('\n')

    def cos_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def remove_stop_word(self, sentence):
        # 文を名詞、形容詞、動詞、副詞のみのリストにする
        part = ['名詞', '動詞', '形容詞', '副詞']
        m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        morphs = m.parse(sentence).split('\n')
        removed = []
        for morph in morphs:
            splited = re.split('[,\t]', morph)
            if len(splited) < 2: continue
            if splited[1] in part:
                removed.append(splited[0])
            # removed.append(splited[0])
        return removed


    def cleaning(self, line):
        line = line.replace('\u3000', '')
        line = line.replace('\n', '')
        return line

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


if __name__ == '__main__':
    verificator = SynopsisSentenceVerificator()
    # verificator.create_doc_embedding_model()
    # verificator.verificate_synopsis_vector_similarity('n0002ei')
    verificator.verificate_synopsis_BoW_simirality('n9859er')
