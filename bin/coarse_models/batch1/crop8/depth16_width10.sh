#!/bin/sh

gpu=2
crop_options=(8)
kernel_options=(2)
depth_options=(16)
width_options=(10)

for i in "${!crop_options[@]}"; do
    cr="${crop_options[i]}"; kr="${kernel_options[i]}"
    for dp in "${depth_options[@]}"; do
      for wd in "${width_options[@]}"; do
        python -u src/train.py \
        --gpu $gpu \
        --coarse_classes \
        --crop_size $cr \
        --kernel_size $kr \
        --depth $dp \
        --width_factor $wd  \
        | tee "logs/model_coarse_all_crop${cr}_kernel${kr}_depth${dp}_width${wd}.log"
    done
  done
done