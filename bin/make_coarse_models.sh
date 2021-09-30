#!/bin/sh

crop_options=(4 8 16)
depth_options=(16 20 24)
width_options=(2 4 6 8)
kernel_options=(2 3 4)

for kr in "${kernel_options[@]}"; do
  for cr in "${crop_options[@]}"; do
    for dp in "${depth_options[@]}"; do
      for wd in "${width_options[@]}"; do
        python -u src/train.py --gpu 7 --coarse_classes --crop_size $cr --depth $dp --width_factor $wd  --kernel_size $kr \
        | tee logs/model_coarse_all_crop${cr}_kernel${kr}_depth${dp}_width${wd}.log
      done
    done
  done
done