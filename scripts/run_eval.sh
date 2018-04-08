#!/usr/bin/env bash
SCRIPTS_PATH=$(dirname $0)

. $SCRIPTS_PATH/args_parse.sh &&

# python3 $SCRIPTS_PATH/../search_based_nmt/decode/t2t_decoder.py \
t2t-decoder \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM\
    --model=$MODEL\
    --hparams_set=$HPARAMS_SET\
    --hparams="$ADDITIONAL_HPARAMS"\
    --output_dir=$TRAIN_DIR \
    --decode_hparams="beam_size=4,alpha=0.5" \
    --decode_from_file="$DATA_DIR/he.$TEST_NAME.txt" \
    --decode_to_file="$DATA_DIR/$RESULT_FILE" \
    --t2t_usr_dir=$T2T_USR_DIR &&

echo &&
echo 'run quality_measurement' &&
python3 $SCRIPTS_PATH/../search_based_nmt/utils/quality_measurement.py \
    --references "$DATA_DIR/en.$TEST_NAME.txt" \
    --sources "$DATA_DIR/he.$TEST_NAME.txt" \
    --hypotheses "$DATA_DIR/$RESULT_FILE" \
    --n $SMOOTH_METHOD
