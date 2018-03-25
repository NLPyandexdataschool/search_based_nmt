from collections import defaultdict


class Translator:
    def __init__(self, path_to_raw_data, file_to_search_name):
        endings = ['.test.txt', '.train.txt', '.dev.txt']
        data = []
        for ending in endings:
            with open(path_to_raw_data + '/en' + ending) as en_file:
                with open(path_to_raw_data + '/he' + ending) as he_file:
                    data.extend(zip(he_file, en_file))
        data = list(set(data))
        self.dict = defaultdict(list)
        with open(file_to_search_name) as handler:
            words_to_search = set([line.strip() for line in handler])
        for he_word, en_word in data:
            if he_word in words_to_search:
                self.dict[he_word.strip()].append(en_word.strip())

    def translate(self, he_word):
        return self.dict[he_word]
