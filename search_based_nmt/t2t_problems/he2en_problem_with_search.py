from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

import tensorflow as tf

import os


def txt_line_split_iterator(txt_path, delimiter='\t'):
    """Iterate through lines of file."""
    with tf.gfile.Open(txt_path) as f:
        for line in f:
            yield [word for word in line.strip().split(delimiter)]


def txt_search_base_iter(dict_path, target_path):
    for splited_dict_line, target in zip(txt_line_split_iterator(dict_path),
                                         text_problems.txt_line_iterator(target_path)):
        nearest = {'nearest1': splited_dict_line[0][:-1], 'nearest2': splited_dict_line[0][1:]}
        yield {'inputs': splited_dict_line[0], 'targets': target, **nearest}


def generate_encoded_samples_for_search_based(sample_generator, encoder):
    for sample in sample_generator:
        sample["inputs"] = encoder.encode(sample["inputs"])
        sample["inputs"].append(text_encoder.EOS_ID)
        sample["targets"] = encoder.encode(sample["targets"])
        sample["targets"].append(text_encoder.EOS_ID)
        for i in range(1, 3):
            key = "nearest" + str(i)
            sample[key] = encoder.encode(sample[key])
            sample[key].append(text_encoder.EOS_ID)
        yield sample


@registry.register_problem('he2en_ws')
class TranslitHeToEnWithSearch(translate.TranslateProblem):
    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def approx_vocab_size(self):
        return 100

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    @property
    def is_generate_per_split(self):
        return True

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        is_train_dataset = dataset_split == problem.DatasetSplit.TRAIN
        dataset_label = 'train' if is_train_dataset else 'dev'
        ext = '.txt'
        he_path = os.path.join(data_dir, 'he.' + dataset_label + ext)
        en_path = os.path.join(data_dir, 'en.' + dataset_label + ext)

        return txt_search_base_iter(he_path, en_path)

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        self.encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        return generate_encoded_samples_for_search_based(generator, self.encoder)
