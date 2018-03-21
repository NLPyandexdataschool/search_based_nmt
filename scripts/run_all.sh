#!/usr/bin/env bash
SCRIPTS_PATH=$(dirname $0)

echo run_datagen &&
$SCRIPTS_PATH/run_datagen.sh $@ &&
echo &&
echo run_train &&
$SCRIPTS_PATH/run_train.sh $@ &&
echo &&
echo run_eval &&
$SCRIPTS_PATH/run_eval.sh $@
