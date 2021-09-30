#!/bin/sh

cl="coarse"
sc="all"
crop_options=(4 8 16)
depth_options=(16 20 24)
width_options=(2 4 6 8)

for cr in "${crop_options[@]}"; do
  for dp in "${depth_options[@]}"; do
    for wd in "${width_options[@]}"; do
  CUDA_VISIBLE_DEVICES=7 python -u src/train.py \
  --${cl}_classes --crop_size $cr --depth $dp --width_factor $wd \
  | tee logs/model_${cl}_${sc}_crop${cr}_depth${dp}_width${wd}.log
    done
  done
done