import copy

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def gru(inputs, hparams, train, name, initial_state=None):
    """Run GRU cell on inputs, assuming they are [batch x time x size]."""

    def dropout_gru_cell():
        return tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(hparams.hidden_size),
                input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

    layers = [dropout_gru_cell() for _ in range(hparams.num_hidden_layers)]
    with tf.variable_scope(name):
        return tf.nn.dynamic_rnn(
                tf.contrib.rnn.MultiRNNCell(layers),
                inputs,
                initial_state=initial_state,
                dtype=tf.float32,
                time_major=False)


def gru_attention_decoder(inputs, hparams, train, name, initial_state, encoder_outputs):
    """Run GRU cell with attention on inputs of shape [batch x time x size]."""

    def dropout_gru_cell():
        return tf.contrib.rnn.DropoutWrapper(
            tf.nn.rnn_cell.GRUCell(hparams.hidden_size),
            input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

    layers = [dropout_gru_cell() for _ in range(hparams.num_hidden_layers)]
    if hparams.attention_mechanism == "luong":
        attention_mechanism_class = tf.contrib.seq2seq.LuongAttention
    elif hparams.attention_mechanism == "bahdanau":
        attention_mechanism_class = tf.contrib.seq2seq.BahdanauAttention
    else:
        raise ValueError("Unknown hparams.attention_mechanism = %s, must be "
                         "luong or bahdanau." % hparams.attention_mechanism)
    attention_mechanism = attention_mechanism_class(hparams.hidden_size, encoder_outputs)

    cell = tf.contrib.seq2seq.AttentionWrapper(
        tf.nn.rnn_cell.MultiRNNCell(layers),
        [attention_mechanism]*hparams.num_heads,
        attention_layer_size=[hparams.attention_layer_size]*hparams.num_heads,
        output_attention=(hparams.output_attention == 1))

    batch_size = common_layers.shape_list(inputs)[0]

    initial_state = cell.zero_state(batch_size, tf.float32).clone(cell_state=initial_state)

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


def gru_bid_encoder(inputs, hparams, train, name):
    """Bidirectional GRU for encoding inputs that are [batch x time x size]."""

    def dropout_gru_cell():
        return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.GRUCell(hparams.hidden_size),
            input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

    with tf.variable_scope(name):
        cell_fw = tf.contrib.rnn.MultiRNNCell(
            [dropout_gru_cell() for _ in range(hparams.num_hidden_layers)])

        cell_bw = tf.contrib.rnn.MultiRNNCell(
            [dropout_gru_cell() for _ in range(hparams.num_hidden_layers)])

        ((encoder_fw_outputs, encoder_bw_outputs),
         (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            dtype=tf.float32,
            time_major=False)

        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
        encoder_states = []

        for i in range(hparams.num_hidden_layers):
            encoder_state = tf.concat(
                values=(encoder_fw_state[i], encoder_bw_state[i]),
                axis=1,
                name="bidirectional_concat")

            encoder_states.append(encoder_state)

        encoder_states = tuple(encoder_states)
        return encoder_outputs, encoder_states


def gru_seq2seq_internal_attention_bid_encoder(inputs, targets, hparams, train):
    """GRU seq2seq model with attention, main step used for training."""
    with tf.variable_scope("gru_seq2seq_attention_bid_encoder"):
        # Flatten inputs.
        inputs = common_layers.flatten4d3d(inputs)
        # GRU encoder.
        encoder_outputs, final_encoder_state = gru_bid_encoder(
            tf.reverse(inputs, axis=[1]), hparams, train, "encoder")
        # GRU decoder with attention
        shifted_targets = common_layers.shift_right(targets)
        hparams_decoder = copy.copy(hparams)
        hparams_decoder.hidden_size = 2 * hparams.hidden_size
        decoder_outputs, _ = gru_attention_decoder(
            common_layers.flatten4d3d(shifted_targets), hparams_decoder, train,
            "decoder", final_encoder_state, encoder_outputs)
        return tf.expand_dims(decoder_outputs, axis=2)


@registry.register_model
class GRUSeq2seqAttentionBidirectionalEncoder(t2t_model.T2TModel):

    def body(self, features):
        # TODO(lukaszkaiser): investigate this issue and repair.
        if self._hparams.initializer == "orthogonal":
            raise ValueError("GRU models fail with orthogonal initializer.")
        train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
        return gru_seq2seq_internal_attention_bid_encoder(
            features.get("inputs"), features["targets"], self._hparams, train)
