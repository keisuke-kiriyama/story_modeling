import os
import json
import MeCab
import logging
from gensim.models import word2vec
import numpy as np
from src.util import settings

class NarouCorpus:
    # story_modeling/data/narou以下にjsonファイルが
    # 格納されているcontentsディレクトリを用意

    def __init__(self):
        novel_contents_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'contents')
        novel_meta_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'meta')
        self.contents_file_paths = [os.path.join(novel_contents_dir_path, file_name) for file_name in os.listdir(novel_contents_dir_path) if not file_name == '.DS_Store']
        self.meta_file_paths = [os.path.join(novel_meta_dir_path, file_name) for file_name in os.listdir(novel_meta_dir_path) if not file_name == '.DS_Store']
        self.morph_set = set()

        # embedding property
        self.wakati_sentences = None
        self.embedding_size = 200
        self.embedding_window = 15
        self.embedding_min_count = 0
        self.embedding_sg = 0

        # init property
        self.data_crensing()
        self.morph_indices = dict((c, i) for i, c in enumerate(self.morph_set))
        self.indices_morph = dict((i, c) for i, c in enumerate(self.morph_set))

        # embedding
        self.embedding_model = self.embedding()

    def load(self, file_path):
        json_file = open(file_path, 'r')
        contents = json.load(json_file)
        json_file.close()
        return contents

    def cleaning(self, line):
        line = line.replace('\u3000', '')
        line = line.replace('\n', '')
        return line

    def wakati(self, line):
        m = MeCab.Tagger('-Owakati')
        wakati = m.parse(line)
        return wakati

    def contents(self, ncode):
        file_path = [path for path in self.contents_file_paths if ncode in path]
        if not len(file_path) == 1:
            print('invalid ncode')
            return
        contents = self.load(file_path[0])['contents']
        for episode_index in range(len(contents)):
            for line_index in range(len(contents[episode_index])):
                contents[episode_index][line_index] = self.cleaning(contents[episode_index][line_index])
        return contents

    def contents_from_file_path(self, file_path):
        if not file_path in self.contents_file_paths:
            print("does not exist file")
            return
        contents = self.load(file_path)['contents']
        for episode_index in range(len(contents)):
            for line_index in range(len(contents[episode_index])):
                contents[episode_index][line_index] = self.cleaning(contents[episode_index][line_index])
        return contents

    def synopsis(self, ncode):
        file_path = [path for path in self.meta_file_paths if ncode in path]
        if not len(file_path) == 1:
            print('invalid ncode')
            return
        synopsis = self.cleaning(self.load(file_path[0])['story'])
        return synopsis

    def synopsis_from_file_path(self, file_path):
        if not file_path in self.meta_file_paths:
            print('does not exist file')
            return
        synopsis = self.cleaning(self.load(file_path)['story'])
        return synopsis


    def data_crensing(self):
        # 分かち書きされた文のリストと、形態素の集合を作成
        wakati_sentences = []
        self.morph_set.add(' ')
        self.morph_set.add('eos')
        for i, contents_file_path in enumerate(self.contents_file_paths):
            print('contents process progress: {}'.format(i / len(self.contents_file_paths)))
            contents = self.load(contents_file_path)['contents']
            for episode in contents:
                for line in episode:
                    line = self.cleaning(line)
                    wakati_line = self.wakati(line).split()
                    for morph in wakati_line:
                        if not morph in self.morph_set:
                            self.morph_set.add(morph)
                    wakati_sentences.append(wakati_line)
        self.wakati_sentences = wakati_sentences

        # メタ情報に使われている単語も形態素集合に追加する
        for i, meta_file_path in enumerate(self.meta_file_paths):
            print('meta process progress: {}'.format(i / len(self.meta_file_paths)))
            meta = self.load(meta_file_path)
            title = meta['title']
            title_morphs = self.wakati(title).split()
            synopsis = meta['story']
            synopsis_morphs = self.wakati(synopsis).split()
            for title_morph in title_morphs:
                if not title_morph in self.morph_set:
                    self.morph_set.add(title_morph)
            for synopsis_morph in synopsis_morphs:
                if not synopsis_morph in self.morph_set:
                    self.morph_set.add(synopsis_morph)


    def embedding(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = word2vec.Word2Vec(self.wakati_sentences,
                                  size = self.embedding_size,
                                  window=self.embedding_window,
                                  min_count=self.embedding_min_count,
                                  sg=self.embedding_sg)
        model_output_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'narou_embedding.model')
        model.save(model_output_path)
        return model

    def contents_to_tensor(self, words_max_length):
        # 小説本文をテンソルに変換する
        # shape=(小説数, 単語数, 単語ベクトルサイズ)
        # 1話目のみのテンソルを作成
        X = np.zeros((len(self.contents_file_paths), words_max_length, self.embedding_size))
        for novel_index, contents in enumerate([self.contents_from_file_path(contents_file_path) for contents_file_path in self.contents_file_paths]):
            episode_one = contents[0]
            for line_index, line in enumerate(episode_one):
                for morph_index, morph in enumerate(self.wakati(line).split()):
                    if morph_index == 0:
                        model_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'narou_embedding.model')
                        model = word2vec.Word2Vec.load(model_path)
                        print(model.__dict__['wv'][morph])




def test_embedding():
    model_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'narou_embedding.model')
    model = word2vec.Word2Vec.load(model_path)
    results = model.wv.most_similar(positive=['部屋'])
    for result in results:
        print(result)


