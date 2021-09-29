#!/bin/sh

#classes=("coarse", "fine")
crop_options=(4 8 16 32)
super_classes=("all")
for cl in "coarse"; do
  for sc in "${super_classes[@]}"; do
    for cs in "${crop_options[@]}"; do
      CUDA_VISIBLE_DEVICES=7 python -u -m src.data.cifar100 --${cl}_classes --crop_size $cs \
      | tee logs/dataset_${cl}_${sc}_crop${cs}.log
    done
  done
done

super_classes=("aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables", "household_electrical_devices", "household_furniture", "insects", "large_carnivores", "large_man - made_outdoor_things", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores", "medium_mammals", "non - insect_invertebrates", "people", "reptiles", "small_mammals", "trees", "vehicles_1", "vehicles_2")
for cl in "fine"; do
  for sc in "${super_classes[@]}"; do
    for cs in "${crop_options[@]}"; do
      CUDA_VISIBLE_DEVICES=7 python -u src/data/cifar100.py --${cl}_classes --crop_size $cs \
      | tee logs/dataset_${cl}_${sc}_crop${cs}.log
    done
  done
done