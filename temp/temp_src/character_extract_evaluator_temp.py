import os
import MeCab
# from src.util import settings
import src.util
from src.bccwj.evaluator.character_extract_evaluator import CharacterExtractEvaluator

def wakati(text):
    m = MeCab.Tagger('-Owakati')
    print(m.parse(text).split())

if __name__ == '__main__':
    file_path = os.path.join(src.util.settings.LITERATURE_DIR_PATH, 'PB59_00489.xml')
    character_extract_evaluator = CharacterExtractEvaluator(file_path)
    print(character_extract_evaluator.raw_text)
    # character_extract_evaluator.eval_extract_character()
    # character_extract_evaluator.strict_eval_extract_character()
    # print(character_extract_evaluator.correct_character_list_flat)
    # print(character_extract_evaluator.correct_character_list_hierarchy)
    # print(character_extract_evaluator.strict_correct_character_list)
    # print(character_extract_evaluator.expectation_character_list)
    # print(character_extract_evaluator.precision)
    # print(character_extract_evaluator.recall)
    # print(character_extract_evaluator.f_measure)
    # print(character_extract_evaluator.strict_precision)
    # print(character_extract_evaluator.strict_recall)
    # print(character_extract_evaluator.strict_f_measure)
    # print(character_extract_evaluator.raw_text)
