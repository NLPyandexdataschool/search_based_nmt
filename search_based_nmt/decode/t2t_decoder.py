import os
import tensorflow as tf

# Dependency imports

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.bin.t2t_decoder import create_decode_hparams, create_hparams, FLAGS
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir


def decode(estimator, hparams, decode_hp):
    from decode_utils import decode_from_file_search_based
    decode_from_file_search_based(estimator, FLAGS.decode_from_file, hparams,
                                  decode_hp, FLAGS.decode_to_file,
                                  checkpoint_path=FLAGS.checkpoint_path)
    if FLAGS.checkpoint_path and FLAGS.keep_timestamp:
        ckpt_time = os.path.getmtime(FLAGS.checkpoint_path + ".index")
        os.utime(FLAGS.decode_to_file, (ckpt_time, ckpt_time))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
    FLAGS.use_tpu = False    # decoding not supported on TPU

    hp = create_hparams()
    decode_hp = create_decode_hparams()

    estimator = trainer_lib.create_estimator(
        FLAGS.model,
        hp,
        t2t_trainer.create_run_config(hp),
        decode_hparams=decode_hp,
        use_tpu=False)

    decode(estimator, hp, decode_hp)


if __name__ == "__main__":
    tf.app.run()
