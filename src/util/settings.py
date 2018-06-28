from os import path, pardir

SETTINGS_PATH = path.abspath(__file__)
UTIL_PATH = path.dirname(SETTINGS_PATH)
SRC_DIR_PATH = path.abspath(path.join(UTIL_PATH, pardir))
PROJECT_ROOT = path.abspath(path.join(SRC_DIR_PATH, pardir))
ANALYSIS_DIR_PATH = path.abspath(path.join(PROJECT_ROOT, 'analysis'))
BCCWJ_ANALYSIS_DIR_PATH = path.abspath(path.join(ANALYSIS_DIR_PATH, 'bccwj'))
CHARACTER_EXTRACT_ANALYSIS = path.abspath(path.join(BCCWJ_ANALYSIS_DIR_PATH, 'character_extract_analysis'))
DATA_DIR_PATH = path.abspath(path.join(PROJECT_ROOT, 'data'))
AOZORA_DATA_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'aozora'))
BCCWJ_DATA_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'bccwj'))
NAROU_DATA_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'narou'))
LITERATURE_DIR_PATH = path.abspath(path.join(BCCWJ_DATA_DIR_PATH, 'Literature'))
TEMP_DIR_PATH = path.abspath(path.join(PROJECT_ROOT, 'temp'))
TEMP_DATA_PATH = path.abspath(path.join(TEMP_DIR_PATH, 'temp_data'))
MODEL_DIR_PATH = path.abspath(path.join(PROJECT_ROOT, 'model'))
NAROU_MODEL_DIR_PATH = path.abspath(path.join(MODEL_DIR_PATH, 'narou'))

