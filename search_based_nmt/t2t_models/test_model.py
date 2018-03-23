from tensor2tensor.utils.t2t_model import T2TModel
from tensor2tensor.utils import registry

import tensorflow as tf
from tensorflow.python.eager import context
from search_based_nmt.search_engine.searcher import Searcher

import six


@registry.register_model("test_sb_model")
class TestModel(T2TModel):
    def body(self, features):
        print(features.keys())
