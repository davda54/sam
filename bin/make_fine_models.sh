#!/bin/sh

crop_options=(16 32)
kernel_options=(4 8)
depth_options=(16 22 28)
width_options=(6 8 10)
super_classes=("food_containers" "large_man-made_outdoor_things" "non-insect_invertebrates")
#super_classes=("aquatic_mammals" "fish" "flowers" "food_containers" "fruit_and_vegetables"
#"household_electrical_devices" "household_furniture" "insects" "large_carnivores" "large_man-made_outdoor_things"
#"large_natural_outdoor_scenes" "large_omnivores_and_herbivores" "medium_mammals" "non-insect_invertebrates"
#"people" "reptiles" "small_mammals" "trees" "vehicles_1" "vehicles_2")


for i in "${!crop_options[@]}"; do
    cr="${crop_options[i]}"; kr="${kernel_options[i]}"
    for dp in "${depth_options[@]}"; do
      for wd in "${width_options[@]}"; do
        for sc in "${super_classes[@]}"; do
          python -u src/train.py \
          --gpu 7 \
          --fine_classes \
          --superclass $sc \
          --crop_size $cr \
          --kernel_size $kr \
          --depth $dp \
          --width_factor $wd  \
          | tee logs/model_fine_${sc}_crop${cr}_kernel${kr}_depth${dp}_width${wd}.log
        done
      done
    done
  done
done