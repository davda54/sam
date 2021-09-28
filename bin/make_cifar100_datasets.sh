#!/bin/sh

for class in fine coarse
do
  for cs in 4 8 16
  do
      for bs in 128
      do
          for th in 2
          do
              CUDA_VISIBLE_DEVICES=7 python -u src/data/cifar100.py \
              --${class}_classes --crop_size $cs --batch_size $bs --threads $th \
              | tee output/log_dataset_CIFAR100_fineFalse_crop${cs}_batch${bs}_threads${th}.log
            done
        done
  done
done