#!/bin/sh

classes=("coarse", "fine")
crop_sizes=(4, 8, 16, 32)
sc="All"
for class in coarse
do
  for cs in "${crop_sizes[@]}"
  do
    CUDA_VISIBLE_DEVICES=7 python -u src/data/cifar100.py \
    --${class}_classes --crop_size $cs --batch_size $bs --threads $th \
    | tee output/log_dataset_${class}_${sc}_crop${cs}_batch${bs}_threads${th}.log
  done
done

for class in coarse
do
  for cs in "${crop_sizes[@]}"
  do
    CUDA_VISIBLE_DEVICES=7 python -u src/data/cifar100.py \
    --${class}_classes --crop_size $cs --batch_size $bs --threads $th \
    | tee output/log_dataset_${class}_class${sc}_crop${cs}_batch${bs}_threads${th}.log
  done
done

superclasses=("aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables", "household_electrical_devices", "household_furniture", "insects", "large_carnivores", "large_man - made_outdoor_things", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores", "medium_mammals", "non - insect_invertebrates", "people", "reptiles", "small_mammals", "trees", "vehicles_1", "vehicles_2")

for class in fine
do
  for sc in "${superclasses[@]}" in
  do
    for cs in "${crop_sizes[@]}"
    do
      CUDA_VISIBLE_DEVICES=7 python -u src/data/cifar100.py \
      --${class}_classes --crop_size $cs --batch_size $bs --threads $th \
      | tee output/log_dataset_${class}_class${sc}_crop${cs}_batch${bs}_threads${th}.log
    done
  done
done