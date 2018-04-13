import os
from collections import defaultdict

path_to_raw_data = '/Users/boyalex/Downloads/shad/nlp/project/proj/search_based_nmt/sb_dir/raw_data'
file_to_search_name = os.path.join(path_to_raw_data, 'he.search.txt')

endings = ['.test.txt', '.train.txt', '.dev.txt']
data = []
for ending in endings:
    en_path = os.path.join(path_to_raw_data, 'en' + ending)
    he_path = os.path.join(path_to_raw_data, 'he' + ending)
    with open(en_path) as en_file:
        with open(he_path) as he_file:
            data.extend(zip(he_file, en_file))
data = list(set(data))

d = defaultdict(list)

with open(file_to_search_name) as handler:
    words_to_search = set(handler.readlines())

words_to_search = set(x.strip() for x in words_to_search)

print (len(data))
hes = set([i[0].strip() for i in data])
print (len(words_to_search))

print ([x for x in words_to_search if x not in hes])