#!/bin/sh

classes=("coarse")
super_classes=("all")
crop_options=(4 8 16 32)

for cl in classes; do
  for sc in super_classes; do
    for cs in "${crop_options[@]}"; do
      CUDA_VISIBLE_DEVICES=7 python -u  src/cifar100.py --${cl}_classes --crop_size $cs \
      | tee logs/dataset_${cl}_${sc}_crop${cs}.log
    done
  done
done
