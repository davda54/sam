#!/bin/sh

crop_options=(4 8 16 32)

for cs in "${crop_options[@]}"; do
  CUDA_VISIBLE_DEVICES=7 python -u  src/cifar100.py --coarse_classes --crop_size $cs \
  | tee logs/dataset_coarse_all_crop${cs}.log
done