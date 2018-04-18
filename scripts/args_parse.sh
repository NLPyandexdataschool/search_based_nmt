#!/bin/bash
SCRIPTS_PATH=$(dirname $0)

export HOME_DIR="$SCRIPTS_PATH/../search_based_nmt"
export DATA_DIR="$HOME_DIR/data/raw_data"
export TMP_DIR="$HOME_DIR/data/t2t_data/tmp"
export TRAIN_DIR="$HOME_DIR/data/t2t_data/train"
export T2T_USR_DIR="$HOME_DIR"

export TRAIN_NAME='train_no_search' # unused
export DEV_NAME='clear_dev' # unused
export TEST_NAME='clear_test'
export SEARCH_NAME='search'

export TABLE_PATH="$HOME_DIR/search_engine/big_table.txt"

export PROBLEM=translit_he_to_en
export MODEL=lstm_seq2seq_attention_bidirectional_encoder
export HPARAMS_SET=lstm_attention
export ADDITIONAL_HPARAMS=""
export BATCH_SIZE=128
export EVAL_FREQUENCY=1000
export KEEP_CHECKPOINT_MAX=20
export TRAIN_STEPS=100000
export SEED=3189

export SMOOTH_METHOD=0


while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --sb|--search-based)
    MODEL=search_based_model
    HPARAMS_SET=search_based_hparams
    PROBLEM=he2en_ws
    shift
    ;;

    --table_path)
    export TABLE_PATH="$2"
    shift
    shift
    ;;

    --data_dir)
    export DATA_DIR="$2"
    shift # past argument
    shift # past value
    ;;

    --tmp_dir)
    export TMP_DIR="$2"
    shift # past argument
    shift # past value
    ;;

    --train_dir)
    export TRAIN_DIR="$2"
    shift # past argument
    shift # past value
    ;;

    --t2t_usr_dir)
    export T2T_USR_DIR="$2"
    shift # past argument
    shift # past value
    ;;

    -r|--result_file)
    export RESULT_FILE="$2"
    shift # past argument
    shift # past value
    ;;

    -p|--problem)
    export PROBLEM="$2"
    shift # past argument
    shift # past value
    ;;

    -m|--model)
    export MODEL="$2"
    shift # past argument
    shift # past value
    ;;

    --params_set)
    export HPARAMS_SET="$2"
    shift # past argument
    shift # past value
    ;;

    --batch_size)
    export BATCH_SIZE="$2"
    shift # past argument
    shift # past value
    ;;

    --additional_params)
    export ADDITIONAL_HPARAMS="$2"
    shift # past argument
    shift # past value
    ;;
    --train_steps)
    export TRAIN_STEPS="$2"
    shift # past argument
    shift # past value
    ;;

    --eval_frequency)
    export EVAL_FREQUENCY="$2"
    shift # past argument
    shift # past value
    ;;

    --keep_checkpoint_max)
    export KEEP_CHECKPOINT_MAX="$2"
    shift # past argument
    shift # past value
    ;;

    -s|--random_seed)
    export SEED="$2"
    shift # past argument
    shift # past value
    ;;

    --smooth_method)
    export SMOOTH_METHOD="$2"
    shift # past argument
    shift # past value
    ;;

    --train_name)
    export TRAIN_NAME="$2"
    shift # past argument
    shift # past value
    ;;

    --dev_name)
    export DEV_NAME="$2"
    shift # past argument
    shift # past value
    ;;

    --test_name)
    export TEST_NAME="$2"
    shift # past argument
    shift # past value
    ;;

    --search_name)
    export SEARCH_NAME="$2"
    shift # past argument
    shift # past value
    ;;

    --generated_data_dir)
    export GENERATED_DATA_DIR="$2"
    shift
    shift
    ;;

    --results_dir)
    export RESULTS_DIR="$2"
    shift
    shift
    ;;

    *)    # unknown option
    echo "unknown option $1"
    shift # past argument
    ;;
esac
done

if [ -z $GENERATED_DATA_DIR ]
then
    export GENERATED_DATA_DIR="$TRAIN_DIR/$TRAIN_NAME"
fi

if [ -z $RESULTS_DIR ]
then
    export RESULTS_DIR="$TRAIN_DIR/results"
fi

if [ -z $RESULT_FILE ]
then
    export RESULT_FILE="$TEST_NAME.results.txt"
fi

# create all folders if not exist
if [ ! -d "$TMP_DIR" ]; then
    mkdir -p "$TMP_DIR"
fi

if [ ! -d "$TRAIN_DIR" ]; then
    mkdir -p "$TRAIN_DIR"
fi

if [ ! -d "$GENERATED_DATA_DIR" ]; then
    mkdir -p "$GENERATED_DATA_DIR"
fi

if [ ! -d "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
fi
