#!/bin/sh

python -u src/cifar100.py --coarse_classes --crop_size $cs |
  tee "logs/dataset_coarse_all_crop${cs}.log"

super_classes=("aquatic_mammals" "fish" "flowers" "food_containers" "fruit_and_vegetables"
  "household_electrical_devices" "household_furniture" "insects" "large_carnivores" "large_man-made_outdoor_things"
  "large_natural_outdoor_scenes" "large_omnivores_and_herbivores" "medium_mammals" "non-insect_invertebrates"
  "people" "reptiles" "small_mammals" "trees" "vehicles_1" "vehicles_2")

for sc in "${super_classes[@]}"; do
    python -u src/cifar100.py --fine_classes --superclass $sc |
      tee "logs/dataset_fine_${sc}.log"
done
