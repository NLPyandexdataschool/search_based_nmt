import numpy as np
import tensorflow as tf
import os
from collections import defaultdict


from tensor2tensor.bin.t2t_decoder import FLAGS
from tensor2tensor.utils.decoding import _get_sorted_inputs, make_input_fn_from_generator
from tensor2tensor.utils.decoding import _decode_input_tensor_to_features_dict, log_decode_results
from tensor2tensor.utils.decoding import _decode_filename
from tensor2tensor.data_generators import text_encoder

from search_based_nmt.search_engine.searcher import Searcher
from search_based_nmt.search_engine.translator import Translator


def decode_from_file_search_based(estimator,
                                  filename,
                                  hparams,
                                  decode_hp,
                                  decode_to_file=None,
                                  checkpoint_path=None):
    """Compute predictions on entries in filename and write them out."""
    if not decode_hp.batch_size:
        decode_hp.batch_size = 32
        tf.logging.info(
                "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)

    problem_id = decode_hp.problem_idx
    # Inputs vocabulary is set to targets if there are no inputs in the problem,
    # e.g., for language models where the inputs are just a prefix of targets.
    has_input = "inputs" in hparams.problems[problem_id].vocabulary
    inputs_vocab_key = "inputs" if has_input else "targets"
    inputs_vocab = hparams.problems[problem_id].vocabulary[inputs_vocab_key]
    targets_vocab = hparams.problems[problem_id].vocabulary["targets"]
    problem_name = FLAGS.problems.split("-")[problem_id]
    tf.logging.info("Performing decoding from a file.")
    sorted_inputs, sorted_keys = _get_sorted_inputs(filename, decode_hp.shards,
                                                    decode_hp.delimiter)
    num_decode_batches = (len(sorted_inputs) - 1) // decode_hp.batch_size + 1

    data_dir = '/'.join(filename.split('/')[:-1])
    table_path = os.path.join(data_dir, '../../search_engine/big_table.txt')
    he_search_path = os.path.join(data_dir, 'he.search.txt')
    en_search_path = os.path.join(data_dir, 'en.search.txt')
    searcher = Searcher(table_path, he_search_path)
    translator = Translator(data_dir, he_search_path)

    def input_fn():
        input_gen = _decode_batch_input_fn_search_based(
                problem_id, num_decode_batches, sorted_inputs, inputs_vocab, targets_vocab,
                decode_hp.batch_size, decode_hp.max_input_size, searcher, translator,
                hparams.problems[problem_id])
        gen_fn = make_input_fn_from_generator(input_gen)
        example = gen_fn()
        return _decode_input_tensor_to_features_dict(example, hparams)

    decodes = []
    result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)
    for result in result_iter:
        if decode_hp.return_beams:
            beam_decodes = []
            beam_scores = []
            output_beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
            scores = None
            if "scores" in result:
                scores = np.split(result["scores"], decode_hp.beam_size, axis=0)
            for k, beam in enumerate(output_beams):
                tf.logging.info("BEAM %d:" % k)
                score = scores and scores[k]
                _, decoded_outputs, _ = log_decode_results(result["inputs"], beam,
                                                           problem_name, None,
                                                           inputs_vocab, targets_vocab)
                beam_decodes.append(decoded_outputs)
                if decode_hp.write_beam_scores:
                    beam_scores.append(score)
            if decode_hp.write_beam_scores:
                decodes.append("\t".join(
                        ["\t".join([d, "%.2f" % s]) for d, s
                         in zip(beam_decodes, beam_scores)]))
            else:
                decodes.append("\t".join(beam_decodes))
        else:
            _, decoded_outputs, _ = log_decode_results(
                    result["inputs"], result["outputs"], problem_name,
                    None, inputs_vocab, targets_vocab)
            decodes.append(decoded_outputs)

    # Reversing the decoded inputs and outputs because they were reversed in
    # _decode_batch_input_fn
    sorted_inputs.reverse()
    decodes.reverse()
    # If decode_to_file was provided use it as the output filename without change
    # (except for adding shard_id if using more shards for decoding).
    # Otherwise, use the input filename plus model, hp, problem, beam, alpha.
    decode_filename = decode_to_file if decode_to_file else filename
    if decode_hp.shards > 1:
        decode_filename += "%.2d" % decode_hp.shard_id
    if not decode_to_file:
        decode_filename = _decode_filename(decode_filename, problem_name, decode_hp)
    tf.logging.info("Writing decodes into %s" % decode_filename)
    outfile = tf.gfile.Open(decode_filename, "w")
    for index in range(len(sorted_inputs)):
        outfile.write("%s%s" % (decodes[sorted_keys[index]], decode_hp.delimiter))


def _decode_batch_input_fn_search_based(problem_id, num_decode_batches, sorted_inputs,
                                        inputs_vocab, targets_vocab, batch_size, max_input_size,
                                        searcher, translator, hparams):
    tf.logging.info(" batch %d" % num_decode_batches)
    # First reverse all the input sentences so that if you're going to get OOMs,
    # you'll see it in the first batch
    sorted_inputs.reverse()
    for b in range(num_decode_batches):
        tf.logging.info("Decoding batch %d" % b)
        batch_length = 0
        batch = defaultdict(list)
        for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
            nearests = searcher.search(inputs, hparams.num_nearest)
            nearest_targets = [translator.translate_random(word)[0] for word in nearests]
            keys = ["inputs"] + hparams.nearest_keys

            # TODO: rewrite this shit
            for key, word in zip(keys, [inputs] + nearests):
                ids = inputs_vocab.encode(word)
                if max_input_size > 0:
                    # Subtract 1 for the EOS_ID.
                    ids = ids[:max_input_size - 1]
                ids.append(text_encoder.EOS_ID)
                batch[key].append(ids)
                if len(ids) > batch_length:
                    batch_length = len(ids)

            for key, word in zip(hparams.nearest_target_keys, nearest_targets):
                ids = targets_vocab.encode(word)
                if max_input_size > 0:
                    ids = ids[:max_input_size - 1]
                ids.append(text_encoder.EOS_ID)
                batch[key].append(ids)
                if len(ids) > batch_length:
                    batch_length = len(ids)

        for key in ["inputs"] + hparams.nearest_keys + hparams.nearest_target_keys:
            for i, ids in enumerate(batch[key]):
                x = ids + [0] * (batch_length - len(ids))
                batch[key][i] = x
        result = {key: np.array(value).astype(np.int32) for key, value in batch.items()}
        result["problem_choice"] = np.array(problem_id).astype(np.int32),
        yield result
