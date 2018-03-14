DATA_DIR="../raw_data"
TRAIN_DIR="../t2t_data/train"
T2T_USR_DIR="../t2t_problem"
RESULT_FILE=${1:-he-to-en.translit.results.txt}

PROBLEM=translit_he_to_en
MODEL=lstm_seq2seq_attention
HPARAMS=lstm_attention

t2t-decoder \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM\
    --model=$MODEL\
    --hparams_set=$HPARAMS\
    --output_dir=$TRAIN_DIR \
    --decode_hparams="beam_size=4,alpha=0.5" \
    --decode_from_file="$DATA_DIR/he.test.txt" \
    --decode_to_file="$DATA_DIR/$RESULT_FILE" \
    --t2t_usr_dir=$T2T_USR_DIR

python3 ../../quality_measurement.py $DATA_DIR/en.test.txt $DATA_DIR/$RESULT_FILE
