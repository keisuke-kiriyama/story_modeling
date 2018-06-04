from os import path, pardir

SRC_DIR_PATH = path.abspath(pardir)
PROJECT_ROOT = path.abspath(path.join(SRC_DIR_PATH, pardir))
DATA_DIR_PATH = path.abspath(path.join(PROJECT_ROOT, 'data'))
TEMP_DIR_PATH = path.abspath(path.join(PROJECT_ROOT, 'temp'))
TEMP_DATA_PATH = path.abspath(path.join(TEMP_DIR_PATH, 'temp_data'))