DATA_DIR="../raw_data"
TMP_DIR="../t2t_data/tmp"
T2T_USR_DIR="../t2t_problem"

PROBLEM=translit_he_to_en

t2t-datagen \
    --data_dir=$DATA_DIR\
    --tmp_dir=$TMP_DIR\
    --problem=$PROBLEM\
    --t2t_usr_dir=$T2T_USR_DIR\
