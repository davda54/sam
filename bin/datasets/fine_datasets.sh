#!/bin/sh

cl='fine'
crop_options=(8 16 24 32)
super_classes=("aquatic_mammals" "fish" "flowers" "food_containers" "fruit_and_vegetables"
  "household_electrical_devices" "household_furniture" "insects" "large_carnivores" "large_man-made_outdoor_things"
  "large_natural_outdoor_scenes" "large_omnivores_and_herbivores" "medium_mammals" "non-insect_invertebrates"
  "people" "reptiles" "small_mammals" "trees" "vehicles_1" "vehicles_2")

for sc in "${super_classes[@]}"; do
  for cs in "${crop_options[@]}"; do
    python -u src/cifar100.py --${cl}_classes --crop_size $cs --superclass $sc |
      tee "logs/dataset_${cl}_${sc}_crop${cs}.log"
  done
done
