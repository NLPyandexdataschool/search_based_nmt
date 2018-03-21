#!/usr/bin/env bash
SCRIPTS_PATH=$(dirname $0)

. $SCRIPTS_PATH/args_parse.sh

t2t-datagen \
    --data_dir=$DATA_DIR\
    --tmp_dir=$TMP_DIR\
    --problem=$PROBLEM\
    --t2t_usr_dir=$T2T_USR_DIR\
