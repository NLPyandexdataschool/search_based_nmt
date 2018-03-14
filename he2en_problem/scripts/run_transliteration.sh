DATA_DIR="../raw_data"
TRAIN_DIR="../t2t_data/train"
TMP_DIR="../t2t_data/tmp"
T2T_USR_DIR="../t2t_problem"

PROBLEM=translit_he_to_en
MODEL=lstm_seq2seq_attention
HPARAMS=lstm_attention
t2t-datagen \
    --data_dir=$DATA_DIR\
    --tmp_dir=$TMP_DIR\
    --problem=$PROBLEM\
    --t2t_usr_dir=$T2T_USR_DIR\

t2t-trainer \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM\
    --model=$MODEL\
    --hparams_set=$HPARAMS\
    --hparams="batch_size=128" \
    --output_dir=$TRAIN_DIR \
    --train_steps=1 \
    --t2t_usr_dir=$T2T_USR_DIR

t2t-decoder \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM\
    --model=$MODEL\
    --hparams_set=$HPARAMS\
    --output_dir=$TRAIN_DIR \
    --decode_hparams="beam_size=4,alpha=0.5" \
    --decode_from_file="$DATA_DIR/he.test.txt" \
    --decode_to_file="$DATA_DIR/he-to-en.translit.joke.txt" \
    --t2t_usr_dir=$T2T_USR_DIR
