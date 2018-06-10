import os
import xml.etree.ElementTree as ET
from statistics import mean, variance
import MeCab
from src.util import settings

class CharacterExtractEvaluator:
    def __init__(self, mxml_file_path):
        self.mxml_file_path = mxml_file_path
        (mxml_tree, raw_text) = self.text_from_mxml_path(mxml_file_path)
        self.mxml_tree = mxml_tree
        self.raw_text = raw_text

    def text_from_mxml_path(self, mxml_file_path):
        tree = ET.parse(mxml_file_path)
        root = tree.getroot()
        raw_text = ""
        for suw in root.iter('SUW'):
            if type(suw.text) == str:
                raw_text += suw.text
        return root, raw_text

    def extract_character_name_from_mxml_tree(self):
        def remove_ruby(element):
            return "".join([i.text for i in element.findall('ruby')])

        character_list = []
        for luw in self.mxml_tree.iter('LUW'):
            if not '名詞-固有名詞-人名' in luw.get('l_pos'): continue
            character_name = {}
            for suw in luw.findall('SUW'):
                suw_text = suw.text if not suw.text == None else remove_ruby(suw)
                if suw.get('pos') == '名詞-固有名詞-人名-姓':
                    character_name['family_name'] = suw_text
                elif suw.get('pos') == '名詞-固有名詞-人名-名':
                    character_name['given_name'] = suw_text
                elif suw.get('pos') == '名詞-固有名詞-人名-一般':
                    character_name['common_name'] = suw_text
            if not character_name in character_list and character_name:
                character_list.append(character_name)
        self.correct_character_list = character_list
        return character_list

    def extract_character_name_with_mecab(self):
        mecab = MeCab.Tagger("-Ochasen")
        node = mecab.parseToNode(self.raw_text)
        character_list = []
        character_name = {}
        is_previous_name = False
        while node:
            morph = node.surface
            feature = node.feature.split(',')
            pos = '-'.join([feature[1], feature[2], feature[3]])
            node = node.next
            if not '固有名詞-人名' in pos:
                if is_previous_name:
                    if not character_name in character_list:
                        character_list.append(character_name)
                    character_name = {}
                is_previous_name = False
                continue
            if pos == '固有名詞-人名-姓':
                character_name['family_name'] = morph
            elif pos == '固有名詞-人名-名':
                character_name['given_name'] = morph
            elif pos == '固有名詞-人名-一般':
                character_name['common_name'] = morph
            is_previous_name = True
        self.expectation_character_list = character_list
        return character_list

    def eval_extract_characer(self):
        character_list_from_tree = self.extract_character_name_from_mxml_tree()
        character_list_with_mecab = self.extract_character_name_with_mecab()
        if not character_list_from_tree or not character_list_with_mecab:
            self.precision = None
            self.recall = None
            self.f_measure = None
            return
        tp, fp = self.positive_expectation_eval(character_list_from_tree, character_list_with_mecab)
        fn = self.negative_expectation_eval(character_list_from_tree, character_list_with_mecab)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        self.precision = precision
        self.recall = recall
        if precision + recall == 0:
            self.f_measure = None
            return
        f_measure = 2 * precision * recall / (precision + recall)
        self.f_measure = f_measure

    def positive_expectation_eval(self, corrects, expectations):
        tp_count = 0.0
        fp_count = 0.0
        for expectation in expectations:
            if expectation in corrects:
                tp_count += 1
            else:
                fp_count += 1
        return tp_count, fp_count

    def negative_expectation_eval(self, corrects, expectations):
        fn_count = 0.0
        for correct in corrects:
            if correct not in expectations:
                fn_count += 1
        return fn_count


def evaluate_character_extraction_with_mecab():
    literatures_dir_path = os.path.join(settings.BCCWJ_DATA_DIR_PATH, 'Literature')
    mxml_file_paths = [os.path.join(literatures_dir_path, i) for i in os.listdir(literatures_dir_path)]
    precisions = []
    recalls = []
    f_measures = []
    items_count = len(mxml_file_paths)
    for i, mxml_file_path in enumerate(mxml_file_paths):
        character_extractor_evaluator = CharacterExtractEvaluator(mxml_file_path)
        character_extractor_evaluator.eval_extract_characer()
        if character_extractor_evaluator.precision:
            precisions.append(character_extractor_evaluator.precision)
        if character_extractor_evaluator.recall:
            recalls.append(character_extractor_evaluator.recall)
        if character_extractor_evaluator.f_measure:
            f_measures.append(character_extractor_evaluator.f_measure)
        completion_rate = i / items_count
        print("completion_rate: {:.4f}".format(completion_rate))
    precision_mean = mean(precisions)
    precision_variance = variance(precisions)
    evaluation_precision_item_count = len(precisions)
    none_precision_count = items_count - evaluation_precision_item_count
    print('precision Mean: ', precision_mean)
    print('precision Variance: ', precision_variance)
    print('num of precisions evaluationed: ', evaluation_precision_item_count)
    print('cannnot precisions evaluation count: ', none_precision_count)
    recall_mean = mean(recalls)
    recall_variance = variance(recalls)
    evaluation_recall_item_count = len(recalls)
    none_recall_count = items_count - evaluation_recall_item_count
    print('recall Mean: ', recall_mean)
    print('recall Variance: ', recall_variance)
    print('num of recall evaluationed: ', evaluation_recall_item_count)
    print('cannnot recall evaluation count: ', none_recall_count)
    f_measure_mean = mean(f_measures)
    f_measure_variance = variance(f_measures)
    evaluation_f_measure_item_count = len(f_measures)
    none_f_measure_count = items_count - evaluation_f_measure_item_count
    print('f-measure Mean: ', f_measure_mean)
    print('f-measure Variance: ', f_measure_variance)
    print('num of f-measure evaluationed: ', evaluation_f_measure_item_count)
    print('cannnot f-measure evaluation count: ', none_f_measure_count)

if __name__ == '__main__':
    evaluate_character_extraction_with_mecab()

