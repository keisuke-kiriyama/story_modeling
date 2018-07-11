import os
import json
import MeCab
import logging
import joblib
from itertools import chain
from gensim.models import word2vec
import numpy as np
from src.util import settings

class NarouCorpusToEmbedding:
    # story_modeling/data/narou以下にjsonファイルが
    # 格納されているcontentsディレクトリを用意

    def __init__(self, is_data_updated=False, is_embed=False, embedding_size=200):
        # is_data_updated
        # データがアップデートされた際に、word2vec学習用のwakati_sentencesの構築と形態素の集合構築を行う
        # is_embed
        # word2vecのモデルを学習し直す

        # paths
        novel_contents_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'contents')
        novel_meta_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'meta')
        self.contents_file_paths = [os.path.join(novel_contents_dir_path, file_name) for file_name in os.listdir(novel_contents_dir_path) if not file_name == '.DS_Store']
        self.meta_file_paths = [os.path.join(novel_meta_dir_path, file_name) for file_name in os.listdir(novel_meta_dir_path) if not file_name == '.DS_Store']
        self.wakati_sentences_file_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'wakati_sentences.txt')
        self.morph_set_file_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'morph_set.txt')
        self.embedding_model_path = os.path.join(settings.NAROU_MODEL_DIR_PATH, 'narou_embedding.model')

        # embedding property
        if is_data_updated:
            self.wakati_sentences = self.create_wakati_sentences()
        elif is_embed:
            self.wakati_sentences = self.load_wakati_sentences()
        else:
            self.wakati_sentences = None
        self.embedding_size = embedding_size
        self.embedding_window = 15
        self.embedding_min_count = 0
        self.embedding_sg = 0

        # set of morph
        self.morph_set = self.create_morph_set() if is_data_updated else self.load_morph_set()
        self.morph_indices = dict((c, i) for i, c in enumerate(self.morph_set))
        self.indices_morph = dict((i, c) for i, c in enumerate(self.morph_set))
        self.vocab_size = len(self.morph_set)

        # embedding
        self.embedding_model = self.embedding() if is_embed else self.load_embedding_model()

        # training data property
        self.contents_length = 1000
        self.synopsis_length = 350

        # data to tensor
        self.X, self.Y = self.data_to_tensor_emb_idx()

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
        m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati')
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

    def create_wakati_sentences(self):
        # 分かち書きされた文のリストと、形態素の集合を作成
        wakati_sentences = []
        wakati_sentences.extend(' ')

        for i, contents_file_path in enumerate(self.contents_file_paths):
            print('contents process progress: {:.3f}'.format(i / len(self.contents_file_paths)))
            contents = list(chain.from_iterable(self.load(contents_file_path)['contents']))
            wakati_lines = [self.wakati(self.cleaning(line)).split() for line in contents]
            wakati_sentences.extend(wakati_lines)

        # タイトルとあらすじに使われている形態素を集合に追加
        # あらすじはWord2Vec学習データに追加
        for i, meta_file_path in enumerate(self.meta_file_paths):
            print('meta process progress: {}'.format(i / len(self.meta_file_paths)))
            meta = self.load(meta_file_path)
            synopsis = meta['story']
            wakati_synopsis = self.wakati(synopsis).split()
            wakati_sentences.append(wakati_synopsis)
        with open(self.wakati_sentences_file_path, 'wb') as f:
            joblib.dump(wakati_sentences, f, compress=3)
        return wakati_sentences

    def load_wakati_sentences(self):
        print('loading wakati_sentences')
        with open(self.wakati_sentences_file_path, 'rb') as f:
            return joblib.load(f)

    def create_morph_set(self):
        morph_set = set()
        morph_set.add(' ')
        morph_set.add('eos')
        for i, sentence in enumerate(self.wakati_sentences):
            print('morph set process progress: {:3f}'.format(i / len(self.wakati_sentences)))
            for morph in sentence:
                if not morph in morph_set:
                    morph_set.add(morph)
        with open(self.morph_set_file_path, 'wb') as f:
            joblib.dump(morph_set, f, compress=3)
        return morph_set

    def load_morph_set(self):
        print('loading morph_set...')
        with open(self.morph_set_file_path, 'rb') as f:
            return joblib.load(f)

    def embedding(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = word2vec.Word2Vec(self.wakati_sentences,
                                  size = self.embedding_size,
                                  window=self.embedding_window,
                                  min_count=self.embedding_min_count,
                                  sg=self.embedding_sg)
        model.save(self.embedding_model_path)
        return model

    def load_embedding_model(self):
        print('loading embedding_model...')
        return word2vec.Word2Vec.load(self.embedding_model_path)

    def data_to_tensor_emb_idx(self):
        # 小説本文とあらすじのデータをテンソルに変換する
        # 小説本文: shape=(小説数, 単語数, 単語ベクトルサイズ)
        # あらすじ: shape=(小説数, 単語数, 語彙サイズ)
        # words_max_length: 使用する単語数
        X = np.zeros((len(self.contents_file_paths), self.contents_length, self.embedding_size), dtype=np.float)
        Y = np.zeros((len(self.contents_file_paths), self.synopsis_length, 1), dtype=np.integer)
        for novel_index, contents_file_path in enumerate(self.contents_file_paths):
            print('data to tensor progress: {:3f}'.format(novel_index / len(self.contents_file_paths)))
            contents = self.contents_from_file_path(contents_file_path)
            # 小説本文は1話目のみのテンソルを作成
            # 本文は分かち書きを揃えるためにlineごとにwakatiを行う
            episode_one_wakati_lines = [self.cleaning(self.wakati(line)) for line in contents[0]]
            contents_morphs = ' '.join(episode_one_wakati_lines).split()
            contents_morphs_to_embed = contents_morphs[0:self.contents_length] if len(contents_morphs) > self.contents_length else self.padding(contents_morphs, self.contents_length)
            for contents_morph_index, contents_morph in enumerate(contents_morphs_to_embed):
                try:
                    X[novel_index][contents_morph_index] = self.embedding_model.__dict__['wv'][contents_morph]
                except:
                    print('[ERROR]error in contents_to_tensor: {}'.format(contents_morph))

            # あらすじはOneHotVector
            ncode = contents_file_path.split('/')[-1].split('.')[0]
            meta_file_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'meta', ncode + '_meta.json')
            synopsis = self.synopsis_from_file_path(meta_file_path)
            synopsis_morphs = self.wakati(synopsis).split()
            synopsis_morphs_to_idx = synopsis_morphs[0:self.synopsis_length] if len(synopsis_morphs) > self.synopsis_length else self.padding(synopsis_morphs, self.synopsis_length)
            for synopsis_morph_index, synopsis_morph in enumerate(synopsis_morphs_to_idx):
                try:
                    Y[novel_index][synopsis_morph_index][0] = self.morph_indices[synopsis_morph]
                except KeyError:
                    print('[KEY ERROR]: {}'.format(synopsis_morph))
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
    # corpus = NarouCorpusToEmbedding()
    # X,Y = corpus.data_to_tensor_emb_idx(1000)
    test_embedding()
