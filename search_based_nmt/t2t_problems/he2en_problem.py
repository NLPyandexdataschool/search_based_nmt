from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import metrics

import os


@registry.register_problem
class TranslitHeToEn(translate.TranslateProblem):
    name = 'he_to_en'

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
        ext = '.txt'
        dataset_label = 'train' if is_train_dataset else 'new_dev'
        dataset_prefix = '' if is_train_dataset else '../'

        he_path = os.path.join(data_dir, dataset_prefix + 'he.' + dataset_label + ext)
        en_path = os.path.join(data_dir, dataset_prefix + 'en.' + dataset_label + ext)

        return text_problems.text2text_txt_iterator(he_path, en_path)
