#!/bin/sh

crop_options=(8 16)
kernel_options=(2 4)
depth_options=(16 22 28)
width_options=(2 6 10)

for i in "${!crop_options[@]}"; do
    cr="${crop_options[i]}"; kr="${kernel_options[i]}"
    for dp in "${depth_options[@]}"; do
      for wd in "${width_options[@]}"; do
        echo "model_coarse_all_crop${cr}_kernel${kr}_depth${dp}_width${wd}" >> permutations
    done
  done
done

crop_options=(16 32)
kernel_options=(4 8)
depth_options=(16 22 28)
width_options=(6 8 10)
super_classes=("aquatic_mammals" "fish" "flowers" "food_containers" "fruit_and_vegetables"
"household_electrical_devices" "household_furniture" "insects" "large_carnivores" "large_man-made_outdoor_things"
"large_natural_outdoor_scenes" "large_omnivores_and_herbivores" "medium_mammals" "non-insect_invertebrates"
"people" "reptiles" "small_mammals" "trees" "vehicles_1" "vehicles_2")


for i in "${!crop_options[@]}"; do
    cr="${crop_options[i]}"; kr="${kernel_options[i]}"
    for dp in "${depth_options[@]}"; do
      for wd in "${width_options[@]}"; do
        for sc in "${super_classes[@]}"; do
          echo "model_fine_${sc}_crop${cr}_kernel${kr}_depth${dp}_width${wd}" >> permutations
      done
    done
  done
done