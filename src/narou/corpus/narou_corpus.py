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
        wakati_sentences.append([' '])

        # タイトルとあらすじに使われている形態素を集合に追加
        # あらすじはWord2Vec学習データに追加
        for i, meta_file_path in enumerate(self.meta_file_paths):
            print('meta process progress: {}'.format(i / len(self.meta_file_paths)))
            meta = self.load(meta_file_path)
            title = meta['title']
            title_morphs = self.wakati(title).split()
            for title_morph in title_morphs:
                if not title_morph in self.morph_set:
                    self.morph_set.add(title_morph)
            synopsis = meta['story']
            synopsis_morphs = self.wakati(synopsis).split()
            wakati_sentences.append(synopsis_morphs)
            for synopsis_morph in synopsis_morphs:
                if not synopsis_morph in self.morph_set:
                    self.morph_set.add(synopsis_morph)
        self.wakati_sentences = wakati_sentences


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

    def data_to_tensor_emb_idx(self, contents_length, synopsis_length=300):
        # 小説本文とあらすじのデータをテンソルに変換する
        # 小説本文: shape=(小説数, 単語数, 単語ベクトルサイズ)
        # あらすじ: shape=(小説数, 単語数, 形態素インデックス: 1)
        # words_max_length: 使用する単語数
        X = np.zeros((len(self.contents_file_paths), contents_length, self.embedding_size), dtype=np.float)
        Y = np.zeros((len(self.contents_file_paths), synopsis_length, 1), dtype=np.float)
        for novel_index, contents_file_path in enumerate(self.contents_file_paths):
            contents = self.contents_from_file_path(contents_file_path)
            # 小説本文は1話目のみのテンソルを作成
            # 本文は分かち書きを揃えるためにlineごとにwakatiを行う
            episode_one_wakati_lines = [self.cleaning(self.wakati(line)) for line in contents[0]]
            contents_morphs = ' '.join(episode_one_wakati_lines).split()
            contents_morphs_to_embed = contents_morphs[0:contents_length] if len(contents_morphs) > contents_length else (self.padding(contents_morphs, contents_length))
            for contents_morph_index, contents_morph in enumerate(contents_morphs_to_embed):
                try:
                    X[novel_index][contents_morph_index] = self.embedding_model.__dict__['wv'][contents_morph]
                except:
                    print('[ERROR]error in contents_to_tensor: {}'.format(contents_morph))

            # あらすじは形態素のインデックスによるテンソル
            ncode = contents_file_path.split('/')[-1].split('.')[0]
            meta_file_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'meta', ncode + '_meta.json')
            synopsis = self.synopsis_from_file_path(meta_file_path)
            synopsis_morphs = self.wakati(synopsis).split()
            synopsis_morphs_to_idx = self.padding(synopsis_morphs, synopsis_length)
            for synopsis_morph_index, synopsis_morph in enumerate(synopsis_morphs_to_idx):
                try:
                    Y[novel_index][synopsis_morph_index][0] = self.morph_indices[synopsis_morph] / len(self.morph_indices)
                except:
                    print('[ERROR]error in synopsis_to_tensor: {}'.format(synopsis_morph))
        return X, Y


    def padding(self, morphs, maxlen):
        for _ in range(maxlen - len(morphs)):
            morphs.append(' ')
        return morphs

def test_embedding():
    model_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'narou_embedding.model')
    model = word2vec.Word2Vec.load(model_path)
    results = model.wv.most_similar(positive=['部屋'])
    for result in results:
        print(result)

if __name__ == '__main__':
    corpus = NarouCorpus()
    X,Y = corpus.data_to_tensor_emb_idx(1000)

