DATA_DIR="../raw_data"
TRAIN_DIR="../t2t_data/train"
T2T_USR_DIR="../t2t_problem"

PROBLEM=translit_he_to_en
MODEL=lstm_seq2seq_attention
HPARAMS=lstm_attention

TRAIN_STEPS=${1:-2000}
SEED=${2:-3189}

t2t-trainer \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM\
    --model=$MODEL\
    --hparams_set=$HPARAMS\
    --hparams="batch_size=128" \
    --output_dir=$TRAIN_DIR \
    --random_seed=$SEED \
    --train_steps=$TRAIN_STEPS \
    --t2t_usr_dir=$T2T_USR_DIR
