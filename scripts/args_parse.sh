#!/bin/bash
SCRIPTS_PATH=$(dirname $0)

HOME_DIR="$SCRIPTS_PATH/../search_based_nmt"
DATA_DIR="$HOME_DIR/data/raw_data"
TMP_DIR="$HOME_DIR/data/t2t_data/tmp"
TRAIN_DIR="$HOME_DIR/data/t2t_data/train"
T2T_USR_DIR="$HOME_DIR"

RESULT_FILE="he-to-en.translit.results.txt"

PROBLEM=translit_he_to_en
MODEL=lstm_seq2seq_attention
HPARAMS=lstm_attention

TRAIN_STEPS=2000
SEED=3189


while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --data_dir)
    DATA_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    --tmp_dir)
    TMP_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    --train_dir)
    TRAIN_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    --t2t_usr_dir)
    T2T_USR_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    -r|--result_file)
    RESULTFILE="$2"
    shift # past argument
    shift # past value
    ;;
    -p|--problem)
    PROBLEM="$2"
    shift # past argument
    shift # past value
    ;;

    -m|--model)
    MODEL="$2"
    shift # past argument
    shift # past value
    ;;

    --params_set)
    HPARAMS="$2"
    shift # past argument
    shift # past value
    ;;
    --train_steps)
    TRAIN_STEPS="$2"
    shift # past argument
    shift # past value
    ;;
    -s|--random_seed)
    SEED="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    echo "unknown option $1"
    shift # past argument
    ;;
esac
done
