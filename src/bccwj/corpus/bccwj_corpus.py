import os
import xml.etree.ElementTree as ET
from gensim.models import word2vec
import logging
import functools

from src.util import settings

class BCCWJCorpus:

    def __init__(self):
        self.mxml_file_paths = self.mxml_file_paths()
        self.wakati_sentences = None
        self.embedding_model = None

    def mxml_file_paths(self):
        literature_dir_path = settings.LITERATURE_DIR_PATH
        file_names = os.listdir(literature_dir_path)
        return list(map(lambda x: os.path.join(literature_dir_path, x), file_names))

    def remove_ruby(self, element):
        removed_text = "".join([i.text for i in element.findall('ruby') if type(i.text) == str])
        return removed_text

    def raw_text_from_mxml_path(self, mxml_file_path):
        tree = ET.parse(mxml_file_path)
        root = tree.getroot()
        raw_text = ""
        for luw in root.iter('LUW'):
            for suw in luw.findall('SUW'):
                suw_text = suw.text if not suw.text == None else self.remove_ruby(suw)
                raw_text += suw_text
        raw_text = raw_text.replace("\u3000", "")
        return raw_text

    def wakati_sentence_list_from_mxml_path(self, mxml_file_path):
        tree = ET.parse(mxml_file_path)
        root = tree.getroot()
        wakati_sentence_list = []
        for sentence in root.iter('sentence'):
            wakati_list = []
            for luw in sentence.findall('LUW'):
                for suw in luw.findall('SUW'):
                    suw_text = suw.text if not suw.text == None else self.remove_ruby(suw)
                    if suw_text == '\u3000': continue
                    wakati_list.append(suw_text)
            wakati_sentence_list.append(" ".join(wakati_list))
        return wakati_sentence_list

    def create_wakati_corpus(self):
        num_of_files = len(self.mxml_file_paths)
        sentences = []
        for i, mxml_file_path in enumerate(self.mxml_file_paths):
            sentences += self.wakati_sentence_list_from_mxml_path(mxml_file_path)
            print('completion rate: ', (i / num_of_files))
        self.wakati_sentences = sentences

    def create_embedding_model(self, size, window, sg=0):
        if not self.wakati_sentences:
            print("wakati_sentences is None")
            return
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = word2vec.Word2Vec(self.wakati_sentences, size=size, window=window, min_count=20, sg=sg)
        self.embedding_model = model
        model.save('./wiki.model')

def test_embedding():
    model = word2vec.Word2Vec.load("./wiki.model")
    results = model.wv.most_similar(positive=['振り向く'])
    for result in results:
        print(result)

if __name__ == '__main__':
    bccwj_corpus = BCCWJCorpus()
    bccwj_corpus.create_wakati_corpus()
    print(bccwj_corpus.wakati_sentences)
    # bccwj_corpus.create_embedding_model(200, 15)
    # test_embedding()

