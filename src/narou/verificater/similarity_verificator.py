import os
import json
from itertools import chain
import re
import MeCab
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from src.util import settings

class SynopsisSentenceVerificator:

    def __init__(self):
        self.novel_contents_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'contents')
        self.novel_meta_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'meta')
        self.contents_file_paths = [os.path.join(self.novel_contents_dir_path, file_name) for file_name in os.listdir(self.novel_contents_dir_path) if not file_name == '.DS_Store']
        self.meta_file_paths = [os.path.join(self.novel_meta_dir_path, file_name) for file_name in os.listdir(self.novel_meta_dir_path) if not file_name == '.DS_Store']
        self.model_output_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'doc2vec.model')
        self.labeled_sentences = None

    def create_doc_embedding_model(self):
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
        print('start to train')
        model = Doc2Vec(dm=0, vector_size=300, window=5, min_count=1, workers=4, epochs=600)
        model.build_vocab(labeled_sentences)
        model.train(labeled_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(self.model_output_path)

    def verificate_synopsis_sentence_similarity(self):
        verificate_contents_file_path = self.contents_file_paths[6]
        ncode = self.ncode_from_contents_file_path(verificate_contents_file_path)
        print("verification ncode: {}".format(ncode))
        verificate_meta_file_path = os.path.join(self.novel_meta_dir_path, ncode+'_meta.json')
        contents_lines = list(chain.from_iterable(self.load(verificate_contents_file_path)['contents']))
        synopsis = self.load(verificate_meta_file_path)['story']
        synopsis_lines = re.split('[。？]', synopsis)
        wakati_contents_lines = [self.cleaning(self.wakati(line)).split() for line in contents_lines]
        wakati_synopsis_lines = [self.cleaning(self.wakati(line)).split() for line in synopsis_lines]

        model = Doc2Vec.load(self.model_output_path)
        for synopsis_idx, synopsis_line in enumerate(wakati_synopsis_lines):
            similarity_dict = {}
            for contents_idx, contents_line in enumerate(wakati_contents_lines):
                similarity_dict[contents_idx] = model.docvecs.similarity_unseen_docs(model,
                                                                                     synopsis_line,
                                                                                     contents_line,
                                                                                     alpha = 1,
                                                                                     min_alpha=0.0001,
                                                                                     steps=5)
            sentence_idx, simirality = max(similarity_dict.items(), key=lambda x: x[1])
            print('-' * 15)
            print('synopsis index: {}'.format(synopsis_idx))
            print(''.join(wakati_synopsis_lines[synopsis_idx]))
            print('similarity: {}'.format(simirality))
            print(''.join(wakati_contents_lines[sentence_idx]))
            print('\n')
        return model

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
    verificator.verificate_synopsis_sentence_similarity()
    # verificator.create_doc_embedding_model()
