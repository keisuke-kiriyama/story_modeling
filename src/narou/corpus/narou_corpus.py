import os
import json
import MeCab
import logging
from gensim.models import word2vec
from src.util import settings

class NarouCorpus:
    # story_modeling/data/narou以下にjsonファイルが
    # 格納されているcontentsディレクトリを用意

    def __init__(self):
        novel_contents_dir_path = os.path.join(settings.NAROU_DATA_DIR_PATH, 'contents')
        print(settings.NAROU_DATA_DIR_PATH)
        self.contents_file_paths = [os.path.join(novel_contents_dir_path, file_name) for file_name in os.listdir(novel_contents_dir_path) if not file_name == '.DS_Store']
        self.wakati_sentences = None
        self.embedding_size = 200
        self.embedding_window = 15
        self.embedding_min_count = 20
        self.embedding_sg = 0
        self.data_crensing()

    def load(self, file_path):
        json_file = open(file_path, 'r')
        contents = json.load(json_file)
        json_file.close()
        return contents

    def cleaning(self, line):
        line = line.replace('\u3000', '')
        return line

    def wakati(self, line):
        m = MeCab.Tagger('-Owakati')
        wakati = m.parse(line)
        return wakati

    def data_crensing(self):
        wakati_sentences = []
        for i, contents_file_path in enumerate(self.contents_file_paths):
            print('progress: {}'.format(i / len(self.contents_file_paths)))
            contents = self.load(contents_file_path)['contents']
            for episode in contents:
                for line in episode:
                    line = self.cleaning(line)
                    wakati_line = self.wakati(line)
                    wakati_sentences.append(wakati_line)
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

if __name__ == '__main__':
    corpus = NarouCorpus()
    corpus.embedding()

