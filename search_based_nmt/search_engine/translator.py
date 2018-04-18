from collections import defaultdict
from numpy.random import choice


class Translator:
    def __init__(self, src_search_name, trg_search_name):
        self._translations = defaultdict(list)
        with open(src_search_name, encoding='utf8') as src_file, \
                open(trg_search_name, encoding='utf8') as trg_file:
            for src, trg in zip(src_file, trg_file):
                self._translations[src.strip()].append(trg.strip())

    def translate_all(self, he_word):
        return self._translations[he_word]

    def translate_random(self, he_word):
        return choice(self._translations[he_word])
