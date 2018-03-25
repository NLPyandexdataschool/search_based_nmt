from tensor2tensor.utils.t2t_model import T2TModel
from tensor2tensor.utils import registry

import tensorflow as tf
from tensorflow.python.eager import context
from search_based_nmt.search_engine.searcher import Searcher
from tensor2tensor.layers import common_layers
from itertools import count
from search_based_nmt.rnn_cells.lstm import LSTMShallowFusionCell

import six


def rnn(inputs, rnn_cell, hparams, train, name, initial_state=None):
    """Run LSTM cell on inputs, assuming they are [batch x time x size]."""

    def dropout_lstm_cell():
        return tf.contrib.rnn.DropoutWrapper(
            rnn_cell,
            input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

    layers = [dropout_lstm_cell() for _ in range(hparams.num_hidden_layers)]
    with tf.variable_scope(name):
        return tf.nn.dynamic_rnn(
            tf.contrib.rnn.MultiRNNCell(layers),
            inputs,
            initial_state=initial_state,
            dtype=tf.float32,
            time_major=False)


def lstm_seq2seq_search_based_attention(inputs, targets, hparams, train, build_storage, storage):
    """LSTM seq2seq search-based model with attention"""
    with tf.variable_scope("lstm_seq2seq_attention", reuse=True):
        # Flatten inputs.
        inputs = common_layers.flatten4d3d(inputs)
        # LSTM encoder.
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hparams.hidden_size)
        encoder_outputs, final_encoder_state = rnn(
            tf.reverse(inputs, axis=[1]), lstm_cell, hparams, train, "encoder")
        # LSTM decoder with attention
        shifted_targets = common_layers.shift_right(targets)
        search_based_rnn_cell = LSTMShallowFusionCell(hparams.hidden_size, build_storage, storage)
        decoder_outputs, _ = lstm_attention_search_based_decoder(
            common_layers.flatten4d3d(shifted_targets),
            search_based_rnn_cell,
            hparams, train, "decoder",
            final_encoder_state, encoder_outputs)
        return tf.expand_dims(decoder_outputs, axis=2)


def lstm_attention_search_based_decoder(
        inputs, rnn_cell, hparams, train, name, initial_state, encoder_outputs):
    """Run LSTM cell with attention on inputs of shape [batch x time x size]."""

    def dropout_lstm_cell():
        return tf.contrib.rnn.DropoutWrapper(
            rnn_cell,
            input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

    layers = [dropout_lstm_cell() for _ in range(hparams.num_hidden_layers)]
    if hparams.attention_mechanism == "luong":
        attention_mechanism_class = tf.contrib.seq2seq.LuongAttention
    elif hparams.attention_mechanism == "bahdanau":
        attention_mechanism_class = tf.contrib.seq2seq.BahdanauAttention
    else:
        raise ValueError("Unknown hparams.attention_mechanism = %s, must be "
                         "luong or bahdanau." % hparams.attention_mechanism)
    attention_mechanism = attention_mechanism_class(
        hparams.hidden_size, encoder_outputs)

    cell = tf.contrib.seq2seq.AttentionWrapper(
        tf.nn.rnn_cell.MultiRNNCell(layers),
        [attention_mechanism]*hparams.num_heads,
        attention_layer_size=[hparams.attention_layer_size]*hparams.num_heads,
        output_attention=(hparams.output_attention == 1))

    batch_size = common_layers.shape_list(inputs)[0]

    initial_state = cell.zero_state(batch_size, tf.float32).clone(
        cell_state=initial_state)

    with tf.variable_scope(name):
        output, state = tf.nn.dynamic_rnn(
            cell,
            inputs,
            initial_state=initial_state,
            dtype=tf.float32,
            time_major=False)

        # For multi-head attention project output back to hidden size
        if hparams.output_attention == 1 and hparams.num_heads > 1:
            output = tf.layers.dense(output, hparams.hidden_size)

        return output, state


@registry.register_model()
class LSTMSearchBased(T2TModel):
    def body(self, features):
        train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
        storage = []
        for i in range(self._hparams.num_nearests):
            lstm_seq2seq_search_based_attention(features['nearest' + str(i)],
                                                features['nearest_target' + str(i)],
                                                self._hparams,
                                                train,
                                                build_storage=True,
                                                storage=storage)

        return lstm_seq2seq_search_based_attention(features['inputs'],
                                                   features['targets'],
                                                   self._hparams,
                                                   train,
                                                   build_storage=False,
                                                   storage=storage)
