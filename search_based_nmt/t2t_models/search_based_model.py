from tensor2tensor.utils.t2t_model import T2TModel, log_info
from tensor2tensor.utils import registry

import tensorflow as tf
from tensorflow.python.eager import context
from tensor2tensor.models.lstm import lstm_attention
from search_based_nmt.search_engine.searcher import Searcher
from tensor2tensor.layers import common_layers
from itertools import count
from search_based_nmt.rnn_cells.lstm import LSTMShallowFusionCell, AttentionWrapperSearchBased

import six


def lstm_bid_encoder(inputs, hparams, train, name):
    """Bidirectional LSTM for encoding inputs that are [batch x time x size]."""

    def dropout_lstm_cell():
        return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.BasicLSTMCell(hparams.hidden_size // 2),
            input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

    with tf.variable_scope(name):
        cell_fw = tf.contrib.rnn.MultiRNNCell(
            [dropout_lstm_cell() for _ in range(hparams.num_hidden_layers)])

        cell_bw = tf.contrib.rnn.MultiRNNCell(
            [dropout_lstm_cell() for _ in range(hparams.num_hidden_layers)])

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
            if isinstance(encoder_fw_state[i], tf.contrib.rnn.LSTMStateTuple):
                encoder_state_c = tf.concat(
                    values=(encoder_fw_state[i].c, encoder_bw_state[i].c),
                    axis=1,
                    name="encoder_fw_state_c")
                encoder_state_h = tf.concat(
                    values=(encoder_fw_state[i].h, encoder_bw_state[i].h),
                    axis=1,
                    name="encoder_fw_state_h")
                encoder_state = tf.contrib.rnn.LSTMStateTuple(
                    c=encoder_state_c, h=encoder_state_h)
            elif isinstance(encoder_fw_state[i], tf.Tensor):
                encoder_state = tf.concat(
                    values=(encoder_fw_state[i], encoder_bw_state[i]),
                    axis=1,
                    name="bidirectional_concat")

            encoder_states.append(encoder_state)

        encoder_states = tuple(encoder_states)
        return encoder_outputs, encoder_states


def lstm_seq2seq_search_based_attention(inputs, targets, hparams, train, build_storage, storage, n):
    """LSTM seq2seq search-based model with attention"""
    with tf.variable_scope("lstm_seq2seq_attention", reuse=tf.AUTO_REUSE):
        # Flatten inputs.
        inputs = common_layers.flatten4d3d(inputs)
        # LSTM encoder.
        encoder_outputs, final_encoder_state = lstm_bid_encoder(inputs, hparams, train, "encoder")
        # LSTM decoder with attention
        shifted_targets = common_layers.shift_right(targets)
        decoder_outputs, p_copy = lstm_attention_search_based_decoder(
            common_layers.flatten4d3d(shifted_targets),
            hparams, train, "decoder",
            final_encoder_state, encoder_outputs,
            build_storage, storage, n)

        if build_storage:
            return tf.expand_dims(decoder_outputs, axis=2)
        else:
            return tf.expand_dims(decoder_outputs, axis=2), p_copy


def lstm_attention_search_based_decoder(inputs, hparams, train, name, initial_state,
                                        encoder_outputs, build_storage, storage, n):
    """Run LSTM cell with attention on inputs of shape [batch x time x size]."""

    def dropout_lstm_cell():
        return tf.contrib.rnn.DropoutWrapper(
            LSTMShallowFusionCell(hparams.hidden_size, build_storage, storage),
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

    if not build_storage:
        p_copy = [tf.TensorArray(tf.float32, size=tf.shape(inputs)[1], dynamic_size=True, name='dzeta_dot_q'),
                  tf.TensorArray(tf.float32, size=tf.shape(inputs)[1], dynamic_size=True, name='1_dzeta')]
    else:
        p_copy = None
    # TODO: add fusion_type in hparams
    cell = AttentionWrapperSearchBased(
        tf.nn.rnn_cell.MultiRNNCell(layers),
        [attention_mechanism]*hparams.num_heads,
        storage=storage, build_storage=build_storage, p_copy=p_copy, start_index=n,
        attention_layer_size=[hparams.attention_layer_size]*hparams.num_heads,
        output_attention=(hparams.output_attention == 1),
        fusion_type=hparams.fusion_type)

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

        return output, p_copy


@registry.register_model("search_based_model")
class LSTMSearchBased(T2TModel):
    def _check_hparams(self):
        fusion_type = self._hparams.get('fusion_type')
        if fusion_type not in ['shallow', 'deep']:
            message = 'Unknown fusion_type: {}'.format(fusion_type)
            if fusion_type is None:
                message += '. Make sure you use search_based_hparams'
            raise ValueError(message)

    def body(self, features):
        self._check_hparams()
        train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
        storage = [tf.TensorArray(tf.float32,
                                  size=(tf.shape(features["inputs"])[1] * self._problem_hparams.num_nearest),
                                  dynamic_size=True, name='c'),
                   tf.TensorArray(tf.float32,
                                  size=(tf.shape(features["inputs"])[1] * self._problem_hparams.num_nearest),
                                  dynamic_size=True, name='z')]

        print('neares_keys', self._problem_hparams.nearest_keys)
        for i, (nearest_key, nearest_target_key) in enumerate(zip(self._problem_hparams.nearest_keys,
                                                                  self._problem_hparams.nearest_target_keys)):

            lstm_seq2seq_search_based_attention(features[nearest_key],
                                                features[nearest_target_key],
                                                self._hparams,
                                                train,
                                                build_storage=True,
                                                storage=storage, n=i*tf.shape(features["inputs"])[1])

        storage_stack = [tf.transpose(s.stack(), [1, 0, 2]) for s in storage]
        with open('tmp_log.txt', 'a') as f:
            print('storage', storage, file=f)
            print('storage_stack', storage_stack, file=f)
        return lstm_seq2seq_search_based_attention(features['inputs'],
                                                   features['targets'],
                                                   self._hparams,
                                                   train,
                                                   build_storage=False,
                                                   storage=storage_stack, n=0)

    def bottom(self, features):
        transformed_features = super().bottom(features)
        print(features.keys())
        # here we can define how to transform nearest_targets
        # now they are transformed with one-hot encoding
        target_modality = self._problem_hparams.target_modality
        vocab_size = target_modality._vocab_size
        print("\n\nvocab_size:", vocab_size, "\n\n")
        with tf.variable_scope(target_modality.name, reuse=True):
            for key in self._problem_hparams.nearest_target_keys:
                log_info("Transforming %s with %s.targets_bottom", key, target_modality.name)
                transformed_features[key] = target_modality.targets_bottom(features[key])
                log_info("Transforming %s with one-hot encoding", key)
                # shape is (bs, max_len, vocab_size)
                transformed_features[key + "_one_hot"] = tf.one_hot(features[key],
                                                                    depth=vocab_size,
                                                                    axis=-1)

        return transformed_features

    def model_fn(self, features):
        """We need this for shallow fusion to change logits."""
        transformed_features = self.bottom(features)

        with tf.variable_scope("body"):
            log_info("Building model body")
            body_out, p_copy = self.body(transformed_features)

            output, losses = self._normalize_body_output(body_out)

            if "training" in losses:
                log_info("Skipping T2TModel top and loss because training loss "
                         "returned from body")
                logits = output
            else:
                dzq = tf.transpose(p_copy[0].stack(), [1, 0, 2])
                inv_dz = tf.transpose(p_copy[1].stack(), [1, 0])

                if self._hparams.fusion_type == 'shallow':
                    y_tilda = tf.concat([features[key + "_one_hot"]
                                         for key in self._problem_hparams.nearest_target_keys],
                                        axis=1)
                    p_tilda = tf.diag_part(tf.tensordot(dzq, y_tilda, axes=[[2], [1]]))
                    logits = inv_dz * self.top(output, features)
                else:
                    logits = self.top(output, features)

                losses["training"] = self.loss(logits, features)
        return logits, losses


@registry.register_hparams
def search_based_hparams():
    hparams = lstm_attention()
    hparams.add_hparam("fusion_type", "deep")
    return hparams
