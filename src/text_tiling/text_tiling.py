# -*- coding: utf-8 -*-

import os
from functools import reduce
import MeCab
from src.util import settings
from src.pre_processing.remove_pseudonym_reading import remove_pseudonym_reading


class TextTiling:
    def __init__(self, input_file_path):
        file = open(input_file_path, 'r', encoding='shift-jis')
        text = file.read()
        splited_text = text.split('\n')
        self.title = splited_text[0]
        self.author = splited_text[1]
        self.body = remove_pseudonym_reading(text)
        self.talken_size = 10
        self.talkens = self.word_talkenize()
        self.block_size = 2
        self.lexical_score_by_adjacent_blocks = []
        self.lexical_score_by_vocabulary_introductions = []
        self.lexical_score_by_lexical_chain = []

    def word_talkenize(self):
        tagger = MeCab.Tagger('-Owakati')
        words = tagger.parse(self.body).split()
        length = len(words)
        return [words[i:i + self.talken_size] for i in range(0, length, self.talken_size)]

    def compare_adjacent_blocks(self):
        for i in range(len(self.talkens)):
            if i == 0:
                self.lexical_score_by_adjacent_blocks.append(0)
                continue
            if i == 1:
                first_block = self.talkens[0]
            else:
                first_block = reduce(lambda x, y: x + y, self.talkens[i - self.block_size : i], [])
            second_block = reduce(lambda x, y: x + y, self.talkens[i : i + self.block_size], [])
            first_block_vector = self.talken_vectorization(first_block)
            second_block_vector = self.talken_vectorization(second_block)
            score = 0
            for key, value in first_block_vector.items():
                score += value * second_block_vector.get(key, 0)
            self.lexical_score_by_adjacent_blocks.append(score)


    def talken_vectorization(self, list):
        vector = {}
        for word in list:
            vector[word] = vector.get(word, 0) + 1
        return vector

if __name__ == '__main__':
    input_file_path = os.path.join(settings.TEMP_DATA_PATH, 'neboke.txt')
    text_tiling = TextTiling(input_file_path=input_file_path)
    text_tiling.compare_adjacent_blocks()
