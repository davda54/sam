#!/bin/sh

gpu=7
crop_options=(16)
kernel_options=(4)
depth_options=(28)
width_options=(6)

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
        | tee "logs/model/coarse/all/crop${cr}/kernel${kr}/depth${dp}/width${wd}/model_coarse_all_crop${cr}_kernel${kr}_depth${dp}_width${wd}.log"
    done
  done
done