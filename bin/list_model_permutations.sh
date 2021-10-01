#!/bin/sh

crop_options=(4 8 16)
depth_options=(4 8 12 16 20 24)
width_options=(2 4 6 8)
kernel_options=(2 3 4)

for cr in "${crop_options[@]}"; do
  for kr in "${kernel_options[@]}"; do
    for wd in "${width_options[@]}"; do
      for dp in "${depth_options[@]}"; do
        echo -e "model_coarse_all_crop${cr}_kernel${kr}_depth${dp}_width${wd}" >> permutations
      done
    done
  done
done

crop_options=(16 32)
depth_options=(12 16 20 24 28)
width_options=(6 8 10 12)
kernel_options=(4 8)
super_classes=("aquatic_mammals" "fish" "flowers" "food_containers" "fruit_and_vegetables"
"household_electrical_devices" "household_furniture" "insects" "large_carnivores" "large_man-made_outdoor_things"
"large_natural_outdoor_scenes" "large_omnivores_and_herbivores" "medium_mammals" "non-insect_invertebrates"
"people" "reptiles" "small_mammals" "trees" "vehicles_1" "vehicles_2")

for cr in "${crop_options[@]}"; do
  for kr in "${kernel_options[@]}"; do
    for dp in "${depth_options[@]}"; do
      for wd in "${width_options[@]}"; do
        for sc in "${super_classes[@]}"; do
          echo -e "model_fine_${sc}_crop${cr}_kernel${kr}_depth${dp}_width${wd}" >> permutations
        done
      done
    done
  done
done