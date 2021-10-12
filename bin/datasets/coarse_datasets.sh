#!/bin/sh

crop_options=(8 16 24 32)

for cs in "${crop_options[@]}"; do
  python -u src/cifar100.py --coarse_classes --crop_size $cs |
    tee "logs/dataset_coarse_all_crop${cs}.log"
done
