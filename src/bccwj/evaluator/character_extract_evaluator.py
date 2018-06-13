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
        self.correct_character_list = []
        self.expectation_character_list = []
        self.strict_correct_character_list = []
        self.strict_expectation_character_list = []
        self.precision = None
        self.recall = None
        self.f_measure = None
        self.strict_precision = None
        self.strict_recall = None
        self.strict_f_measure = None
        self.extract_character_name_from_mxml_tree()
        self.extract_character_name_with_mecab()

    def remove_ruby(self, element):
        removed_text = "".join([i.text for i in element.findall('ruby') if type(i.text) == str])
        return removed_text

    def text_from_mxml_path(self, mxml_file_path):
        tree = ET.parse(mxml_file_path)
        root = tree.getroot()
        raw_text = ""
        for luw in root.iter('LUW'):
            for suw in luw.findall('SUW'):
                suw_text = suw.text if not suw.text == None else self.remove_ruby(suw)
                raw_text += suw_text
        return root, raw_text

    def extract_character_name_from_mxml_tree(self):
        # mxmlツリーから姓名を区別して登場人物名を抽出する
        character_list = []
        strict_character_list = []
        for luw in self.mxml_tree.iter('LUW'):
            if not '名詞-固有名詞-人名' in luw.get('l_pos'): continue
            character_name = ''
            character_name_dict = {}
            for suw in luw.findall('SUW'):
                suw_text = suw.text if not suw.text == None else self.remove_ruby(suw)
                if '名詞-固有名詞-人名' in suw.get('pos'):
                    character_name += suw_text
                if suw.get('pos') == '名詞-固有名詞-人名-姓':
                    character_name_dict['family_name'] = suw_text
                elif suw.get('pos') == '名詞-固有名詞-人名-名':
                    character_name_dict['given_name'] = suw_text
                elif suw.get('pos') == '名詞-固有名詞-人名-一般':
                    character_name_dict['common_name'] = suw_text
            if not character_name in character_list:
                character_list.append(character_name)
            if not character_name_dict in strict_character_list and character_name_dict:
                strict_character_list.append(character_name_dict)
        self.correct_character_list = character_list
        self.strict_correct_character_list = strict_character_list

    def extract_character_name_with_mecab(self):
        # MeCabを用いて姓名を区別して登場人物名を抽出する
        # mecab = MeCab.Tagger("-Ochasen")
        mecab = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
        node = mecab.parseToNode(self.raw_text)
        character_list = []
        strict_character_list = []
        character_name = {}
        is_previous_name = False
        while node:
            morph = node.surface
            feature = node.feature.split(',')
            pos = '-'.join([feature[1], feature[2], feature[3]])
            node = node.next
            if not '固有名詞-人名' in pos:
                if is_previous_name:
                    if not character_name in strict_character_list:
                        strict_character_list.append(character_name)
                    character_name = {}
                is_previous_name = False
                continue
            elif not morph in character_list:
                character_list.append(morph)
            if pos == '固有名詞-人名-姓':
                character_name['family_name'] = morph
            elif pos == '固有名詞-人名-名':
                character_name['given_name'] = morph
            elif pos == '固有名詞-人名-一般':
                character_name['common_name'] = morph
            is_previous_name = True
        self.expectation_character_list = character_list
        self.strict_expectation_character_list = strict_character_list

    def eval_extract_character(self):
        # 姓名の区別をせず登場人物抽出の評価を行う
        if not self.correct_character_list or not self.expectation_character_list:
            self.precision = None
            self.recall = None
            self.f_measure = None
            return
        tp, fp = self.positive_expectation_eval(self.correct_character_list, self.expectation_character_list)
        fn = self.negative_expectation_eval(self.correct_character_list, self.expectation_character_list)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        self.precision = precision
        self.recall = recall
        if precision + recall == 0:
            self.strict_f_measure = None
            return

        f_measure = 2 * precision * recall / (precision + recall)
        self.f_measure = f_measure

    def strict_eval_extract_character(self):
        # 姓名も区別して登場人物抽出の評価を行う
        if not self.strict_correct_character_list or not self.strict_expectation_character_list:
            self.strict_precision = None
            self.strict_recall = None
            self.strict_f_measure = None
            return
        tp, fp = self.positive_expectation_eval(self.strict_correct_character_list, self.strict_expectation_character_list)
        fn = self.negative_expectation_eval(self.strict_correct_character_list, self.strict_expectation_character_list)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        self.strict_precision = precision
        self.strict_recall = recall
        if precision + recall == 0:
            self.strict_f_measure = None
            return
        f_measure = 2 * precision * recall / (precision + recall)
        self.strict_f_measure = f_measure

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
    strict_precisions = []
    strict_recalls = []
    strict_f_measures = []
    items_count = len(mxml_file_paths)
    output_file_path = os.path.join(settings.ANALYSIS_DIR_PATH, 'neolog-d.txt')
    with open(output_file_path, 'w') as file:
        for i, mxml_file_path in enumerate(mxml_file_paths):
            character_extractor_evaluator = CharacterExtractEvaluator(mxml_file_path)
            character_extractor_evaluator.eval_extract_character()
            character_extractor_evaluator.strict_eval_extract_character()
            file.write(character_extractor_evaluator.mxml_file_path + '\n')
            file.write(str(character_extractor_evaluator.correct_character_list) + '\n')
            file.write(str(character_extractor_evaluator.expectation_character_list) + '\n\n')
            if character_extractor_evaluator.precision:
                precisions.append(character_extractor_evaluator.precision)
            if character_extractor_evaluator.recall:
                recalls.append(character_extractor_evaluator.recall)
            if character_extractor_evaluator.f_measure:
                f_measures.append(character_extractor_evaluator.f_measure)
            if character_extractor_evaluator.strict_precision:
                strict_precisions.append(character_extractor_evaluator.strict_precision)
            if character_extractor_evaluator.strict_recall:
                strict_recalls.append(character_extractor_evaluator.strict_recall)
            if character_extractor_evaluator.strict_f_measure:
                strict_f_measures.append(character_extractor_evaluator.strict_f_measure)
            completion_rate = i / items_count
            print("completion_rate: {:.4f}".format(completion_rate))
        print('-------------------------------------------')
        print('Evaluation')
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
        print('-------------------------------------------')
        print('Strict Evaluation')
        strict_precision_mean = mean(strict_precisions)
        strict_precision_variance = variance(strict_precisions)
        strict_evaluation_precision_item_count = len(strict_precisions)
        strict_none_precision_count = items_count - strict_evaluation_precision_item_count
        print('precision Mean: ', strict_precision_mean)
        print('precision Variance: ', strict_precision_variance)
        print('num of precisions evaluationed: ', strict_evaluation_precision_item_count)
        print('cannnot precisions evaluation count: ', strict_none_precision_count)
        strict_recall_mean = mean(strict_recalls)
        strict_recall_variance = variance(strict_recalls)
        strict_evaluation_recall_item_count = len(strict_recalls)
        strict_none_recall_count = items_count - strict_evaluation_recall_item_count
        print('recall Mean: ', strict_recall_mean)
        print('recall Variance: ', strict_recall_variance)
        print('num of recall evaluationed: ', strict_evaluation_recall_item_count)
        print('cannnot recall evaluation count: ', strict_none_recall_count)
        strict_f_measure_mean = mean(strict_f_measures)
        strict_f_measure_variance = variance(strict_f_measures)
        strict_evaluation_f_measure_item_count = len(strict_f_measures)
        strict_none_f_measure_count = items_count - strict_evaluation_f_measure_item_count
        print('f-measure Mean: ', strict_f_measure_mean)
        print('f-measure Variance: ', strict_f_measure_variance)
        print('num of f-measure evaluationed: ', strict_evaluation_f_measure_item_count)
        print('cannnot f-measure evaluation count: ', strict_none_f_measure_count)
        print('-------------------------------------------')
        file.write('-------------------------------------------\n')
        file.write('Evaluation\n')
        file.write('precision Mean: ' + str(precision_mean) + '\n')
        file.write('precision Variance: ' + str(precision_variance) + '\n')
        file.write('recall Mean: ' + str(recall_mean) + '\n')
        file.write('recall Variance: ' + str(recall_variance) + '\n')
        file.write('f-measure Mean: ' + str(f_measure_mean) + '\n')
        file.write('f-measure Variance: ' + str(f_measure_variance) + '\n')


def character_extract_error_analysis(file_name):
    mxml_file_path = os.path.join(settings.LITERATURE_DIR_PATH, file_name)
    character_extracter_evaluator = CharacterExtractEvaluator(mxml_file_path)
    character_extracter_evaluator.strict_eval_extract_character()
    output_dir_path = os.path.join(settings.CHARACTER_EXTRACT_ANALYSIS, file_name.split('.')[0])
    if not os.path.exists(output_dir_path): os.mkdir(output_dir_path)
    correct_character_output_file_path = os.path.join(output_dir_path, 'correct.txt')
    with open(correct_character_output_file_path, 'w') as f:
        for character in character_extracter_evaluator.strict_correct_character_list:
            f.write(str(character) + "\n")
    expectation_character_output_file_path = os.path.join(output_dir_path, 'expectation.txt')
    with open(expectation_character_output_file_path, 'w') as f:
        for character in character_extracter_evaluator.strict_expectation_character_list:
            f.write(str(character) + "\n")
        f.write("f-measure: " + str(character_extracter_evaluator.strict_f_measure) + '\n')
        f.write("precision: " + str(character_extracter_evaluator.strict_precision) + '\n')
        f.write("recall: " + str(character_extracter_evaluator.strict_recall) + '\n')
    morph_output_file_path = os.path.join(output_dir_path, 'morph.txt')
    m = MeCab.Tagger('-Ochasen')
    parsed = m.parse(character_extracter_evaluator.raw_text)
    with open(morph_output_file_path, 'w') as f:
        for line in parsed.split('\n'):
            elements = line.split('\t')
            if not len(elements) == 6: continue
            morph_pos = elements[0] + " " + elements[3] + "\n"
            f.write(morph_pos)


def all_file_character_extract_error_anaylysis():
    # analysys/bccwj/characer_extract_analysisに正解、予測、形態素解析結果を示す
    all_file_name = os.listdir(settings.LITERATURE_DIR_PATH)
    for file_name in all_file_name:
        character_extract_error_analysis(file_name)

if __name__ == '__main__':
    evaluate_character_extraction_with_mecab()
    # all_file_character_extract_error_anaylysis()

    # file_path = os.path.join(settings.LITERATURE_DIR_PATH, 'PB39_00045.xml')
    # extractor = CharacterExtractEvaluator(file_path)
    # print(extractor.correct_character_list)
