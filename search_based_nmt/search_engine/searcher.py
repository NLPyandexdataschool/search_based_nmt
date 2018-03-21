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
        return list(set(words))

    def process_table(self, table_file_name, n_nearest=10):
        words = self.get_words()
        with open(table_file_name, 'a+') as handler:
            for i, word in enumerate(words):
                heap = []
                for test_word in words:
                    item = (-self.distance(word, test_word), test_word)
                    if item in heap:
                        continue
                    if len(heap) < n_nearest:
                        heappush(heap, item)
                    else:
                        heappushpop(heap, item)
                handler.write(' '.join([word] + list(reversed([heappop(heap)[1] for _ in range(n_nearest)]))) + '\n')
                print("{} %".format(100 * (i + 1) / len(words)))

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
