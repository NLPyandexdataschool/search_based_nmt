from collections import defaultdict
from numpy.random import choice
import os


class Translator:
    def __init__(self, path_to_raw_data, file_to_search_name=None):
        endings = ['.test.txt', '.train.txt', '.dev.txt']
        data = []
        for ending in endings:
            en_path = os.path.join(path_to_raw_data, 'en' + ending)
            he_path = os.path.join(path_to_raw_data, 'he' + ending)
            with open(en_path) as en_file:
                with open(he_path) as he_file:
                    data.extend([
                        (x.strip(), y.strip())
                        for x, y in zip(he_file, en_file)
                    ])

        data = list(set(data))
        self.dict = defaultdict(list)

        if file_to_search_name is not None:
            with open(file_to_search_name) as handler:
                words_to_search = set([l.strip() for l in handler])
            for he_word, en_word in data:
                if he_word in words_to_search:
                    self.dict[he_word].append(en_word)
        else:
            for he_word, en_word in data:
                self.dict[he_word].append(en_word)

    def translate_all(self, he_word):
        return self.dict[he_word]

    def translate_random(self, he_word):
        return choice(self.dict[he_word])
