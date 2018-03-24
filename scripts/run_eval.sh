#!/usr/bin/env bash
SCRIPTS_PATH=$(dirname $0)

. $SCRIPTS_PATH/args_parse.sh &&

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

echo
echo 'run quality_measurement'
python3 $SCRIPTS_PATH/../search_based_nmt/utils/quality_measurement.py $DATA_DIR/en.test.txt $DATA_DIR/$RESULT_FILE
