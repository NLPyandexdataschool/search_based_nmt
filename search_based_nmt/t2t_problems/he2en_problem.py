from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

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
        dataset_label = os.getenv('TRAIN_NAME') if is_train_dataset else os.getenv('DEV_NAME')
        original_data_dir = os.getenv('DATA_DIR')

        he_path = os.path.join(original_data_dir, 'he.' + dataset_label + ext)
        en_path = os.path.join(original_data_dir, 'en.' + dataset_label + ext)

        return text_problems.text2text_txt_iterator(he_path, en_path)
