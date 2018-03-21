from tensor2tensor.utils.t2t_model import T2TModel
from tensor2tensor.utils import registry

import tensorflow as tf
from tensorflow.python.eager import context
from search_based_nmt.search_engine.searcher import Searcher

import six


@registry.register_model("test_sb_model")
class TestModel(T2TModel):
    def body(self, features):
        # in case we can evaluate tf.Tensor we can get string like this
        encoder = self._problem_hparams.vocabulary["inputs"]
        letter = first_input[0, 0, 0, 0]
        print(encoder.decode(letter))
