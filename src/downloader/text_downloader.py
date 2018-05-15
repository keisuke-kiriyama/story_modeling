# -*- coding: utf-8 -*-
import requests
import zipfile
import io
import os
import logging
from time import sleep
from src.util import settings

def download_zip(url, output_file_path):
    sleep(2)
    try:
        print('downloading: {}'.format(url))
        r = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(r.content)) as zip:
            zip.extractall(output_file_path)
    except:
       error_log(output_file_path)

def error_log(output_file_path):
    logger = logging.getLogger('FailedDownload')
    logger.setLevel(10)
    fh = logging.FileHandler(os.path.join(settings.TEMP_DIR_PATH, 'log', 'new_pseudonym_text_download_error.log'))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    logger.log(40, 'failed to download: {}'.format(output_file_path))
