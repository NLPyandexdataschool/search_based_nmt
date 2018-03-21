. ./args_parse.sh

if ! [ -z $TRAIN_DIR ]
then
    cd $TRAIN_DIR
    ls | grep -v ^gitkeep | grep -v ^gitignore | xargs wc # rm -rf
else
    echo '$TRAIN_DIR is empty'
fi
