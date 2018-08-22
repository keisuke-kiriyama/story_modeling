import os
import sys
import joblib
import numpy as np
from rouge import Rouge

from src.narou.corpus.narou_corpus import NarouCorpus
from src.narou.corpus.embedding_and_cossim_data import EmbeddingAndCosSimData
from src.util import settings

class EmbeddingAndBinClassifiedSentenceData:
    """
    付与されたスコア(cos類似度)が高い順に文を１文ずつあらすじに採用していき
    最も高いROUGEスコアが高い文数をOPTのあらすじにする
    その時の採用された文と採用されなかった文のスコアの境が重要度の閾値となる
    error_line_indexes: データ作成時にエラーがでた文のインデックス
    X: 文中の単語ベクトルの平均ベクトル
    Y: 採用された文を正例、採用されなかった文を負例
    Y_score: 本文とあらすじ文との類似度が最も高い値
    threshold: 最後に採用された文に付与されたスコア
    rouge: 最も値が高かった際のROUGEスコア
    {
        ncode:
         {
         error_line_indexes: np.array,
         X: np.array,
         Y: np.array,
         Y_score: np.array,
         threshold: float,
         rouge:
            {
            f: float
            r: float,
            p: float
            }
        }
    }
    """

    def __init__(self):

        # Data
        self.embedding_and_cossim_data = EmbeddingAndCosSimData()
        self.corpus = self.embedding_and_cossim_data.corpus

        # PATH
        self.embedding_and_bin_classified_sentence_data_dict_path = os.path.join(settings.NAROU_MODEL_DIR_PATH,
                                                                                 'embedding_and_bin_classified_sentence',
                                                                                 'embedding_and_bin_classified_sentence.txt')
        # PROPERTY
        self.max_sentence_count = 30
        self.input_vector_size = 200

    def load_data_dict(self):
        print('loading data dict...')
        with open(self.embedding_and_bin_classified_sentence_data_dict_path, 'rb') as f:
            data_dict = joblib.load(f)
        return data_dict

    def create_per_novel_data_dict(self, ncode, emb_cossim_data):
        """
        各小説のデータを作成する
        :param emb_cossim_data: dict
        :return: dict
        {
         error_line_indexes: np.array,
         X: np.array,
         Y: np.array,
         threshold: float,
         rouge:
            {
            f: float,
            r: float,
            p: float
            }
        }
        """
        print('processing Ncode: {}'.format(ncode))
        rouge = Rouge()

        contents_lines = self.corpus.get_contents_lines(ncode)
        removed_contents_lines = np.array(self.corpus.remove_error_line_indexes_from_contents_lines(contents_lines,
                                                                                           emb_cossim_data['error_line_indexes']))

        ref = self.corpus.wakati(''.join(self.corpus.get_synopsis_lines(ncode)))
        if removed_contents_lines.size == 0 or not ref:
            print('contents or synopsis is empty')
            return None
        max_sentence_count = min(self.max_sentence_count, len(emb_cossim_data['X']))
        high_score_line_indexes = np.argsort(-emb_cossim_data['Y'])[:max_sentence_count]
        high_score_slices = [high_score_line_indexes[:i + 1] for i in range(max_sentence_count)]
        synopsises = [self.corpus.wakati(''.join(removed_contents_lines[indices])) for indices in high_score_slices]
        try:
            scores = [rouge.get_scores(hyps=hyp, refs=ref, avg=False)[0]['rouge-1'] for hyp in synopsises]
        except RecursionError as err:
            print(err)
            return None

        # max_fscore_index + 1が採用する文数に相当
        max_f_score_index = np.argmax([score['f'] for score in scores])

        binary_sentences = np.zeros(len(emb_cossim_data['Y']), dtype=np.int)
        pos_indexes = high_score_line_indexes[:max_f_score_index + 1]
        binary_sentences[pos_indexes] = 1

        per_novel_data_dict = dict()
        per_novel_data_dict['error_line_indexes'] = emb_cossim_data['error_line_indexes']
        per_novel_data_dict['X'] = emb_cossim_data['X']
        per_novel_data_dict['Y'] = binary_sentences
        per_novel_data_dict['Y_score'] = emb_cossim_data['Y']
        # 採用されなかった文で最も値が高かったもののスコアを閾値とする
        per_novel_data_dict['threshold'] = emb_cossim_data['Y'][high_score_line_indexes[min(max_f_score_index + 1, max_sentence_count - 1)]]
        per_novel_data_dict['rouge'] = {'f': scores[max_f_score_index]['f'],
                                        'r': scores[max_f_score_index]['r'],
                                        'p': scores[max_f_score_index]['p']}
        return per_novel_data_dict

    def create_data_dict(self):
        emb_cossim_data_dict = self.embedding_and_cossim_data.non_seq_data_dict_emb_cossim(tensor_refresh=False)
        data_dict = dict()
        sys.setrecursionlimit(20000)
        for file_index, (ncode, data) in enumerate(emb_cossim_data_dict.items()):
            print('[INFO] num of processed novel count: {}'.format(file_index))
            per_novel_dict = self.create_per_novel_data_dict(ncode=ncode, emb_cossim_data=data)
            if per_novel_dict is None:
                continue
            data_dict[ncode] = per_novel_dict
        print('saving data_dict...')
        with open(self.embedding_and_bin_classified_sentence_data_dict_path, 'wb') as f:
            joblib.dump(data_dict, f, compress=3)
        return data_dict

    def embedding_and_bin_classified_sentence_data_dict(self, data_refresh=False):
        """
        データを新たに作るか、ロードするか判別し、データを返す
        """
        is_data_dict_exist = os.path.isfile(self.embedding_and_bin_classified_sentence_data_dict_path)
        if is_data_dict_exist and not data_refresh:
            data_dict = self.load_data_dict()
        else:
            data_dict = self.create_data_dict()
        return data_dict

def max_synopsis_sentence_count():
    corpus = NarouCorpus()
    max_ncode = ''
    max_count = 0
    for meta_file_path in corpus.meta_file_paths:
        synopsis_lines = corpus.load(meta_file_path)['story']
        if len(synopsis_lines) > max_count:
            max_ncode = corpus.ncode_from_meta_file_path(meta_file_path)
            max_count = len(synopsis_lines)

    synopsis_lines = corpus.get_synopsis_lines('n0194er')
    print(synopsis_lines)
    return max_ncode, max_count

if __name__ == '__main__':
    supplier = EmbeddingAndBinClassifiedSentenceData()
    supplier.create_data_dict()

