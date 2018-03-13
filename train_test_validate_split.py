import os
import shutil
from sklearn.model_selection import train_test_split


def train_test_validate_split(train_size=0.5, test_size=0.4, val_size=0.1,
                              read_path='he-en', write_path='split',
                              random_seed=None):
    if train_size + test_size + val_size != 1:
        raise Exception('Train, test and val sizes must sum to 1!')

    languages = ['en', 'he', 'hewv']
    types = ['dev', 'train', 'test']
    file_names = [
        '.'.join([lang, cur_type, 'txt'])
        for lang in languages
        for cur_type in types
    ]
    # Сортируем по reversed именам, чтобы добавлять dev, test и train из разных файлов в одном порядке
    file_names = sorted(file_names, key=(lambda x: ''.join(reversed(x))))

    en, he, hewv = [], [], []
    for file_name in file_names:
        with open(os.path.join(read_path, file_name)) as handler:
            if file_name[:2] == 'en':
                en.extend([line for line in handler])
            elif file_name[2:4] == 'wv':
                hewv.extend([line for line in handler])
            else:
                he.extend([line for line in handler])
    # Откусываем val
    en_tmp, en_val, he_tmp, he_val, hewv_tmp, hewv_val = train_test_split(
        en, he, hewv, test_size=val_size, random_state=random_seed
    )

    # Осташееся делим на train и test
    en_train, en_test, he_train, he_test, hewv_train, hewv_test = train_test_split(
        en_tmp, he_tmp, hewv_tmp,
        test_size=test_size / (test_size + train_size),
        random_state=random_seed
    )

    # Очищаем и создаем пустую директорию
    if write_path in os.listdir():
        shutil.rmtree(write_path)
    os.mkdir(write_path)

    write_file_names = [
        '.'.join([lang, cur_type, 'txt'])
        for lang in languages
        for cur_type in ['train', 'test', 'val']
    ]
    write_list = [en_train, en_test, en_val, he_train, he_test, he_val, hewv_train, hewv_test, hewv_val]
    for file_name, write_values in zip(write_file_names, write_list):
        with open(os.path.join(write_path, file_name), 'a+') as handler:
            for value in write_values:
                handler.write(value)


train_test_validate_split()
