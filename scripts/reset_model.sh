. ./args_parse.sh

cd $TRAIN_DIR
ls | grep -v ^gitkeep | grep -v ^gitignore | xargs rm -rf
