from os import path, pardir

SETTINGS_PATH = path.abspath(__file__)
UTIL_PATH = path.dirname(SETTINGS_PATH)
SRC_DIR_PATH = path.abspath(path.join(UTIL_PATH, pardir))
PROJECT_ROOT = path.abspath(path.join(SRC_DIR_PATH, pardir))
DATA_DIR_PATH = path.abspath(path.join(PROJECT_ROOT, 'data'))
AOZORA_DATA_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'aozora'))
BCCWJ_DATA_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'bccwj'))
TEMP_DIR_PATH = path.abspath(path.join(PROJECT_ROOT, 'temp'))
TEMP_DATA_PATH = path.abspath(path.join(TEMP_DIR_PATH, 'temp_data'))
