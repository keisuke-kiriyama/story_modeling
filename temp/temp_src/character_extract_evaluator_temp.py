import os
import MeCab
from src.util import settings
from src.bccwj.evaluator.character_extract_evaluator import CharacterExtractEvaluator

def wakati(text):
    m = MeCab.Tagger('-Owakati')
    print(m.parse(text).split())

if __name__ == '__main__':
    file_path = os.path.join(settings.LITERATURE_DIR_PATH, 'PB29_00363.xml')
    character_extract_evaluator = CharacterExtractEvaluator(file_path)
    print(character_extract_evaluator.raw_text)
    # wakati(character_extract_evaluator.raw_text)
