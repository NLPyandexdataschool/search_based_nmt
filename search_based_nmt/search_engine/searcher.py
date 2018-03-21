from nltk.metrics.distance import edit_distance
from heapq import heappush, heappushpop, heappop


class SearcherException(Exception):
    pass


class Searcher:
    def __init__(self, file_names):
        '''
        Argument file_names is a list of files to search names.
        '''
        self.file_names = file_names

    def distance(self, first_word, second_word):
        return edit_distance(first_word, second_word)

    def get_words(self):
        words = []
        for file_name in self.file_names:
            with open(file_name) as handler:
                words.extend([line.strip() for line in handler])
        return words

    def process_table(self, table_file_name):
        pass

    def search(self, word, n_nearest=10):
        '''
        Search n nearest words to given word.
        '''
        heap = []
        for file_name in self.file_names:
            with open(file_name) as handler:
                for line in handler:
                    item = (-self.distance(word.strip(), line.strip()), line.strip())
                    if item in heap:
                        continue
                    if len(heap) < n_nearest:
                        heappush(heap, item)
                    else:
                        heappushpop(heap, item)
        if len(heap) != n_nearest:
            raise SearcherException("Wrong result length!")
        return list(reversed([heappop(heap)[1] for _ in range(n_nearest)]))
