# -*- coding: utf-8 -*-

import os
import math
from functools import reduce
import MeCab
import numpy as np
from statistics import mean, stdev
from matplotlib import pyplot
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
        self.token_size = 20
        # self.tokens = self.word_tokenize()
        self.tokens = self.word_tokenize_by_sentence()
        self.block_size = 2
        self.lexical_score_by_adjacent_blocks = []
        self.lexical_score_by_vocabulary_introductions = []
        self.lexical_score_by_lexical_chain = []
        self.deep_score_by_adjacent_blocks = []
        self.deep_score_by_vocabulary_introductions = []
        self.deep_score_by_lexical_chain = []
        self.token_scene_comb = []

    def word_tokenize(self):
        tagger = MeCab.Tagger('-Owakati')
        words = tagger.parse(self.body).split()
        length = len(words)
        return [words[i:i + self.token_size] for i in range(0, length, self.token_size)]

    def word_tokenize_by_sentence(self):
        sentences = self.body.split('。')
        tagger = MeCab.Tagger('-Owakati')
        return list(map(lambda x: tagger.parse(x).split(), sentences))

    def talken_vectorization(self, list):
        vector = {}
        for word in list:
            vector[word] = vector.get(word, 0) + 1
        return vector

    def compare_adjacent_blocks(self):
        for i in range(len(self.tokens)):
            if i == 0 or i == len(self.tokens):
                self.lexical_score_by_adjacent_blocks.append(0)
                continue
            elif i == 1:
                first_block = self.tokens[0]
                second_block = reduce(lambda x, y: x + y, self.tokens[i: i + self.block_size], [])
            elif i == len(self.tokens) - 1:
                first_block = reduce(lambda x, y: x + y, self.tokens[i - self.block_size : i], [])
                second_block = self.tokens[len(self.tokens) - 1]
            else:
                first_block = reduce(lambda x, y: x + y, self.tokens[i - self.block_size : i], [])
                second_block = reduce(lambda x, y: x + y, self.tokens[i : i + self.block_size], [])
            first_block_vector = self.talken_vectorization(first_block)
            second_block_vector = self.talken_vectorization(second_block)
            score = 0
            first_block_vector_size = 0
            for key, value in first_block_vector.items():
                score += value * second_block_vector.get(key, 0)
                first_block_vector_size += value ** 2
            second_block_vector_size = reduce(lambda x,y: x + y, second_block_vector.values())
            lexical_score = float(score) / math.sqrt(first_block_vector_size * second_block_vector_size)
            self.lexical_score_by_adjacent_blocks.append(lexical_score)

    def determinig_deep_score_by_adjacent_blocks(self):
        for i, lexical_score in enumerate(self.lexical_score_by_adjacent_blocks):
            left_top = 0
            right_top = 0
            for l in self.lexical_score_by_adjacent_blocks[:i]:
                if l > left_top:
                    left_top = l
            for r in self.lexical_score_by_adjacent_blocks[i+1:]:
                if r > right_top:
                    right_top = r
            self.deep_score_by_adjacent_blocks.append((left_top - lexical_score) + (right_top - lexical_score))

    def smoothing(self, window_size, repeat):
        width = int(window_size/2)
        for _ in range(repeat):
            for i, deep_score in enumerate(self.deep_score_by_adjacent_blocks):
                try:
                    left_score = self.deep_score_by_adjacent_blocks[i - width]
                    right_score = self.deep_score_by_adjacent_blocks[i + width]
                    self.deep_score_by_adjacent_blocks[i] = (left_score + deep_score + right_score) / 3
                except:
                    pass

    def boundary_identification(self):
        average = mean(self.deep_score_by_adjacent_blocks)
        st_dev = stdev(self.deep_score_by_adjacent_blocks)
        threshold = average - st_dev/2
        scene_number = 0
        for i, deep_score in enumerate(self.deep_score_by_adjacent_blocks):
            if deep_score > threshold:
                scene_number += 1
            self.token_scene_comb.append([''.join(self.tokens[i]), scene_number])


    def visualization(self, value_sequence):
        x_axis = np.arange(len(value_sequence))
        pyplot.xlabel("token index")
        pyplot.ylabel("lexical score")
        pyplot.plot(x_axis, value_sequence)
        pyplot.axhline(0.31347, ls=':')
        pyplot.show()


if __name__ == '__main__':
    input_file_path = os.path.join(settings.TEMP_DATA_PATH, 'neboke.txt')
    text_tiling = TextTiling(input_file_path=input_file_path)
    text_tiling.compare_adjacent_blocks()
    text_tiling.determinig_deep_score_by_adjacent_blocks()
    text_tiling.smoothing(window_size=1,repeat=2)
    text_tiling.visualization(value_sequence=text_tiling.deep_score_by_adjacent_blocks)
    text_tiling.boundary_identification()
    for line in text_tiling.token_scene_comb:
        print(line)
