from src.util import settings
import os

def create_literature_bibliography():
    bibliography_path = os.path.join(settings.BCCWJ_DATA_DIR_PATH, 'Bibliography.txt')
    literature_lines = []
    with open(bibliography_path, 'r') as input_file:
        for line in input_file.readlines():
            if line.split('\t')[10] == '9 文学':
                literature_lines.append(line)
    bibliography = "".join(literature_lines)
    output_file_path = os.path.join(settings.BCCWJ_DATA_DIR_PATH, 'Literature_Bibliography.txt')
    with open(output_file_path, 'w') as output_file:
        output_file.write(bibliography)


if __name__ == '__main__':
    create_literature_bibliography()