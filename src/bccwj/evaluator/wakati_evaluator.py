from src.util import settings
import os
import xml.etree.ElementTree as ET

class WakatiEvaluator:

    def __init__(self, mxml_file_path):
        self.mxml_file_path = mxml_file_path
        self.raw_text, self.suw_list = self.parse_mxml_path(mxml_file_path)

    def remove_ruby(self, element):
        removed_text = "".join([i.text for i in element.findall('ruby') if type(i.text) == str])
        return removed_text

    def parse_mxml_path(self, mxml_file_path):
        tree = ET.parse(mxml_file_path)
        root = tree.getroot()
        raw_text = ""
        suw_list = []
        for luw in root.iter('LUW'):
            for suw in luw.findall('SUW'):
                suw_text = suw.text if not suw.text == None else self.remove_ruby(suw)
                raw_text += suw_text
                suw_list.append(suw_text)
        return raw_text, suw_list




if __name__ == '__main__':
    mxml_file_path = os.path.join(settings.LITERATURE_DIR_PATH, 'PB19_00003.xml')
    wakati_evaluator = WakatiEvaluator(mxml_file_path)
    print(wakati_evaluator.suw_list)
