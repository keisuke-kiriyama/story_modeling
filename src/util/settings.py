from os import path, pardir

SETTINGS_PATH = path.abspath(path.dirname(__file__))
SRC_DIR_PATH = path.abspath(path.join(SETTINGS_PATH, pardir))
PROJECT_ROOT = path.abspath(path.join(SRC_DIR_PATH, pardir))
