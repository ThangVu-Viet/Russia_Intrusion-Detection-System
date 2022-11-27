#!/bin/bash

model_list=('svm-rbf' 'svm-linear' 'tree')
activation_array=('seg_relu' 'ELU' 'LeakyReLU' 'relu')
file_array=('case_1_test_data_norm' 'case_2_test_data_norm' 'case_3_test_data_norm')

# shellcheck disable=SC2068
for activation in ${activation_array[@]}
  do
    # shellcheck disable=SC1073
    for file_test in ${file_array[@]}
      do
        python tools/dl-predict.py  \
        --mode_weight "model-save" \
        --save_result "./runs/submit" \
        --test_path "./dataset/$file_test.data" \
        -v "$activation-$file_test-model-save-v2.0" \
        --model_path "./runs/training/full-data-v2.0/model-$activation-full-data-norm/$activation-full-data-v2.0/model-save/model-model-$activation-full-data-norm-$activation-full-data-v2.0.h5" \
        -name "model-$activation-$file_test-v2.0"
      done
  done

# shellcheck disable=SC2068
for model in ${model_list[@]}
do
  for file_test in ${file_array[@]}
    do
      python tools/ml-predict.py  \
      --save_result "./runs/submit" \
      --test_path "./dataset/$file_test.data" \
      -v "$model-$file_test-v2.0" \
      --model_path "./runs/training/full-data-v2.0/$model/$model-full-data-norm-v2.0/model-save/$model-$model-full-data-norm-v2.0.sav" \
      -name "model-$model-$file_test-v2.0"
    done
done

