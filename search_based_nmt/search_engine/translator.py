from collections import defaultdict


class Translator:
    def __init__(self, path):
        endings = ['.test.txt', '.train.txt', '.dev.txt']
        data = []
        for ending in endings:
            with open(path + '/en' + ending) as en_file:
                with open(path + '/he' + ending) as he_file:
                    data.extend(zip(he_file, en_file))
        data = list(set(data))
        self.dict = defaultdict(list)
        for he_word, en_word in data:
            self.dict[he_word.strip()].append(en_word.strip())

    def translate(self, he_word):
        return self.dict[he_word]
