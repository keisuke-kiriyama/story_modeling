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
        model = Doc2Vec(dm=1, vector_size=300, window=3, min_count=5, workers=4, epochs=200)
        model.build_vocab(labeled_sentences)
        model.train(labeled_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model_output_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'doc2vec.model')
        model.save(model_output_path)

    def verificate_synopsis_sentence_similarity(self):
        print('loading data...')
        verificate_contents_file_path = self.contents_file_paths[6]
        ncode = self.ncode_from_contents_file_path(verificate_contents_file_path)
        print("verification ncode: {}".format(ncode))
        verificate_meta_file_path = os.path.join(self.novel_meta_dir_path, ncode+'_meta.json')
        contents_lines = list(chain.from_iterable(self.load(verificate_contents_file_path)['contents']))
        synopsis = self.load(verificate_meta_file_path)['story']
        synopsis_lines = re.split('[。？]', synopsis)
        wakati_contents_lines = [self.cleaning(self.wakati(line)).split() for line in contents_lines]
        wakati_synopsis_lines = [self.cleaning(self.wakati(line)).split() for line in synopsis_lines]
        contents_labeled_sentences = [LabeledSentence(words, tags = [str(i)]) for i, words in enumerate(wakati_contents_lines)]
        synopsis_labeled_sentences = [LabeledSentence(words, tags=['synopsis_' + str(i)]) for i, words in enumerate(wakati_synopsis_lines)]
        labeled_sentences = contents_labeled_sentences + synopsis_labeled_sentences
        self.labeled_sentences = labeled_sentences
        print('start to train')
        model = Doc2Vec(dm=1, vector_size=300, window=3, min_count=5, workers=4, epochs=200)
        model.build_vocab(labeled_sentences)
        model.train(labeled_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model_output_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'doc2vec.model')
        model.save(model_output_path)

        # あらすじ各文に類似する文を出力
        for i, synopsis_sentence in enumerate(synopsis_labeled_sentences):
            print('synopsis sentence {}'.format(i))
            print(''.join(synopsis_sentence.words))
            print('-' * 15)
            similar_sentence = model.docvecs.most_similar(synopsis_sentence.tags, topn=30)
            similar_tags = [sentence[0] for sentence in similar_sentence]
            similarity = [sentence[1] for sentence in similar_sentence]
            sentences = list(filter(lambda sentence: sentence.tags[0] in similar_tags,  self.labeled_sentences))
            for sentence, similarity in zip(sentences, similarity):
                print(str(round(similarity, 5)) + ' ' + ''.join(sentence.words))
            return
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
    # verificator.verificate_synopsis_sentence_similarity()
    verificator.create_doc_embedding_model()
