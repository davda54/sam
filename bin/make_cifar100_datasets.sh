#!/bin/sh

#classes=("coarse", "fine")
crop_sizes=(4 8 16 32)
superclasses="all"
for class in coarse; do
  for sc in "${superclasses[@]}"; do
    for cs in "${crop_sizes[@]}"; do
      CUDA_VISIBLE_DEVICES=7 python -u -m src.data.cifar100 --${class}_classes --crop_size $cs \
      | tee logs/log_dataset_${class}_${sc}_crop${cs}.log
    done
  done
done

superclasses=("aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables", "household_electrical_devices", "household_furniture", "insects", "large_carnivores", "large_man - made_outdoor_things", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores", "medium_mammals", "non - insect_invertebrates", "people", "reptiles", "small_mammals", "trees", "vehicles_1", "vehicles_2")
for class in fine; do
  for sc in "${superclasses[@]}"; do
    for cs in "${crop_sizes[@]}"; do
      CUDA_VISIBLE_DEVICES=7 python -u src/data/cifar100.py --${class}_classes --crop_size $cs \
      | tee logs/log_dataset_${class}_${sc}_crop${cs}.log
    done
  done
done