DATA_DIR="../raw_data"
TRAIN_DIR="../t2t_data/train"
T2T_USR_DIR="../t2t_problem"

PROBLEM=translit_he_to_en
MODEL=lstm_seq2seq_attention
HPARAMS=lstm_attention

t2t-trainer \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM\
    --model=$MODEL\
    --hparams_set=$HPARAMS\
    --hparams="batch_size=128" \
    --output_dir=$TRAIN_DIR \
    --train_steps=1000 \
    --t2t_usr_dir=$T2T_USR_DIR
