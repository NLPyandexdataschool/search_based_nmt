#!/usr/bin/env bash
SCRIPTS_PATH=$(dirname $0)

. $SCRIPTS_PATH/args_parse.sh

t2t-trainer \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM\
    --model=$MODEL\
    --hparams_set=$HPARAMS_SET\
    --hparams="batch_size=$BATCH_SIZE,$ADDITIONAL_HPARAMS" \
    --output_dir=$TRAIN_DIR \
    --random_seed=$SEED \
    --train_steps=$TRAIN_STEPS \
    --t2t_usr_dir=$T2T_USR_DIR
