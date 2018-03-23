from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.utils import optimize
from tensor2tensor.utils import data_reader
from tensor2tensor.data_generators.problem import skip_random_fraction, pad_batch
from tensor2tensor.data_generators.problem import _are_shapes_fully_defined, _summarize_features

import tensorflow as tf
import six

import os


def txt_line_split_iterator(txt_path, delimiter='\t'):
    """Iterate through lines of file."""
    with tf.gfile.Open(txt_path) as f:
        for line in f:
            yield [word for word in line.strip().split(delimiter)]


def txt_search_base_iter(dict_path, target_path):
    for splited_dict_line, target in zip(txt_line_split_iterator(dict_path),
                                         text_problems.txt_line_iterator(target_path)):
        nearest = {'nearest1': splited_dict_line[0], 'nearest2': splited_dict_line[0][1:]}
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
        encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        return generate_encoded_samples_for_search_based(generator, encoder)

    def feature_encoders(self, data_dir):
        encoders = super().feature_encoders(data_dir)
        encoders["nearest1"] = encoders["inputs"]
        return encoders

    def hparams(self, defaults, unused_model_hparams):
        super().hparams(defaults, unused_model_hparams)
        nearest_vocab_size = self._encoders["nearest1"].vocab_size
        defaults.input_modality["nearest1"] = (registry.Modalities.SYMBOL, nearest_vocab_size)

    def preprocess_example(self, example, mode, hparams):
        result = super().preprocess_example(example, mode, hparams)
        result["nearest1"] = result["nearest1"][:hparams.max_input_seq_length]
        return result

    def example_reading_spec(self):
        data_fields = {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.VarLenFeature(tf.int64),
            "nearest1": tf.VarLenFeature(tf.int64)
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def input_fn(self,
                 mode,
                 hparams,
                 data_dir=None,
                 params=None,
                 config=None,
                 dataset_kwargs=None):
        """Builds input pipeline for problem.

        Args:
            mode: tf.estimator.ModeKeys
            hparams: HParams, model hparams
            data_dir: str, data directory; if None, will use hparams.data_dir
            params: dict, may include "batch_size"
            config: RunConfig; should have the data_parallelism attribute if not using
                TPU
            dataset_kwargs: dict, if passed, will pass as kwargs to self.dataset
                method when called

        Returns:
            (features_dict<str name, Tensor feature>, Tensor targets)
        """
        partition_id, num_partitions = self._dataset_partition(mode, config)

        is_training = mode == tf.estimator.ModeKeys.TRAIN
        if config and config.use_tpu:
            num_threads = 64
        else:
            num_threads = 4 if is_training else 1

        max_length = self.max_length(hparams)

        def tpu_valid_size(example):
            return data_reader.example_valid_size(
                example,
                hparams.min_length,
                max_length
            )

        def gpu_valid_size(example):
            drop_long_sequences = is_training or hparams.eval_drop_long_sequences
            return data_reader.example_valid_size(
                example,
                hparams.min_length,
                max_length
                if drop_long_sequences else 10**9
            )

        def define_shapes(example):
            batch_size = config and config.use_tpu and params["batch_size"]
            return standardize_shapes(example, batch_size=batch_size)

        # Read and preprocess
        data_dir = data_dir or hparams.data_dir

        dataset_kwargs = dataset_kwargs or {}
        dataset_kwargs.update({
                "mode": mode,
                "data_dir": data_dir,
                "num_threads": num_threads,
                "hparams": hparams,
                "partition_id": partition_id,
                "num_partitions": num_partitions,
        })

        dataset = self.dataset(**dataset_kwargs)
        if is_training:
            # Repeat and skip a random number of records
            dataset = dataset.repeat()
            data_files = tf.contrib.slim.parallel_reader.get_data_files(
                    self.filepattern(data_dir, mode))
            #    In continuous_train_and_eval when switching between train and
            #    eval, this input_fn method gets called multiple times and it
            #    would give you the exact same samples from the last call
            #    (because the Graph seed is set). So this skip gives you some
            #    shuffling.
            dataset = skip_random_fraction(dataset, data_files[0])

        dataset = dataset.map(
                data_reader.cast_int64_to_int32, num_parallel_calls=num_threads)

        if self.batch_size_means_tokens:
            batch_size_means_tokens = True
        else:
            if _are_shapes_fully_defined(dataset.output_shapes):
                batch_size_means_tokens = False
            else:
                tf.logging.warning(
                        "Shapes are not fully defined. Assuming batch_size means tokens. "
                        "Override batch_size_means_tokens() "
                        "in your problem subclass if this is undesired behavior.")
                batch_size_means_tokens = True

        # Batching
        if not batch_size_means_tokens:
            # Batch size means examples per datashard.
            if config and config.use_tpu:
                # on TPU, we use params["batch_size"], which specifies the number of
                # examples across all datashards
                batch_size = params["batch_size"]
                dataset = dataset.apply(
                        tf.contrib.data.batch_and_drop_remainder(batch_size))
            else:
                num_shards = (config and config.data_parallelism.n) or 1
                batch_size = hparams.batch_size * num_shards
                dataset = dataset.batch(batch_size)
        else:
            # batch_size means tokens per datashard
            if config and config.use_tpu:
                dataset = dataset.filter(tpu_valid_size)
                padded_shapes = self._pad_for_tpu(dataset.output_shapes, hparams)
                # on TPU, we use params["batch_size"], which specifies the number of
                # examples across all datashards
                batch_size = params["batch_size"]
                dataset = dataset.apply(
                        tf.contrib.data.padded_batch_and_drop_remainder(
                                batch_size, padded_shapes))
            else:
                # On GPU, bucket by length
                dataset = dataset.filter(gpu_valid_size)
                batching_scheme = data_reader.hparams_to_batching_scheme(
                        hparams,
                        shard_multiplier=(config and config.data_parallelism.n) or 1,
                        length_multiplier=self.get_hparams().batch_size_multiplier)
                if hparams.use_fixed_batch_size:
                    # Here    batch_size really means examples per datashard.
                    batching_scheme["batch_sizes"] = [hparams.batch_size]
                    batching_scheme["boundaries"] = []
                dataset = data_reader.bucket_by_sequence_length(
                        dataset, data_reader.example_length, batching_scheme["boundaries"],
                        batching_scheme["batch_sizes"])

                if not is_training:

                    def _pad_batch(features):
                        if not config or config.data_parallelism.n <= 1:
                            return features
                        tf.logging.warn(
                                "Padding the batch to ensure that remainder eval batches have "
                                "a batch size divisible by the number of data shards. This may "
                                "lead to incorrect metrics for non-zero-padded features, e.g. "
                                "images. Use a single datashard (i.e. 1 GPU) in that case.")
                        return pad_batch(features, config.data_parallelism.n)

                    dataset = dataset.map(_pad_batch, num_parallel_calls=num_threads)

        dataset = dataset.map(define_shapes, num_parallel_calls=num_threads)
        dataset = dataset.prefetch(2)
        features = dataset.make_one_shot_iterator().get_next()
        if not config or not config.use_tpu:
            _summarize_features(features, (config and config.data_parallelism.n) or 1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            features["infer_targets"] = features["targets"]
            features["targets"] = None
            # This is because of a bug in the Estimator that short-circuits prediction
            # if it doesn't see a QueueRunner. DummyQueueRunner implements the
            # minimal expected interface but does nothing.
            tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_reader.DummyQueueRunner())

        return features, features["targets"]

    def serving_input_fn(self, hparams):
        """Input fn for serving export, starting from serialized example."""
        mode = tf.estimator.ModeKeys.PREDICT
        serialized_example = tf.placeholder(
                dtype=tf.string, shape=[None], name="serialized_example")
        dataset = tf.data.Dataset.from_tensor_slices(serialized_example)
        dataset = dataset.map(self.decode_example)
        dataset = dataset.map(lambda ex: self.preprocess_example(ex, mode, hparams))
        dataset = dataset.map(data_reader.cast_int64_to_int32)
        dataset = dataset.padded_batch(1000, dataset.output_shapes)
        dataset = dataset.map(standardize_shapes)
        features = tf.contrib.data.get_single_element(dataset)

        if self.has_inputs:
            features.pop("targets", None)

        return tf.estimator.export.ServingInputReceiver(
                features=features, receiver_tensors=serialized_example)


def standardize_shapes(features, batch_size=None):
    """Set the right shapes for the features."""

    for fname in ["inputs", "targets", "nearest1"]:
        if fname not in features:
            continue

        f = features[fname]
        while len(f.get_shape()) < 4:
            f = tf.expand_dims(f, axis=-1)

        features[fname] = f

    if batch_size:
        # Ensure batch size is set on all features
        for _, t in six.iteritems(features):
            shape = t.get_shape().as_list()
            shape[0] = batch_size
            t.set_shape(t.get_shape().merge_with(shape))
            # Assert shapes are fully known
            t.get_shape().assert_is_fully_defined()

    return features
