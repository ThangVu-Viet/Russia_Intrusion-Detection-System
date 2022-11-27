#!/bin/bash
index_array=('seg_relu' 'ELU' 'LeakyReLU' 'relu')
# shellcheck disable=SC2068
for activation in ${index_array[@]}
do
  python tools/dl-train.py \
  -train "./dataset/full_data_norm.data" \
  -test "./dataset/full_data_test_norm.data" \
  --result_path "./runs/results/full-data-v2.0" \
  --training_path "./runs/training/full-data-v2.0" \
  -ep 1000 -bsize 64 -verbose 2 -activation_block "$activation" \
  -status_ckpt True -status_early_stop True \
  -name "model-$activation-full-data-norm" \
  -v "$activation-full-data-v2.0" \
  -test_size 0.2
done

model_list=('svm-rbf' 'svm-linear' 'tree')
# shellcheck disable=SC2068
for model in ${model_list[@]}
  do
    python tools/ml-train.py \
    -train "./dataset/full_data_norm.data" \
    -test "./dataset/full_data_test_norm.data" \
    --result_path "./runs/results/full-data-v2.0" \
    --training_path "./runs/training/full-data-v2.0" \
    -name "$model" -v "$model-full-data-norm-v2.0"
  done