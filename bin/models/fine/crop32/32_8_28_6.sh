#!/bin/sh

gpu=7
cr=32
kr=8
dp=28
wd=6
super_classes=("aquatic_mammals" "fish" "fruit_and_vegetables" "household_electrical_devices" "household_furniture" "insects" "large_carnivores" "large_natural_outdoor_scenes" "large_omnivores_and_herbivores" "medium_mammals" "reptiles" "small_mammals" "trees"  "vehicles_2")

for sc in "${super_classes[@]}"; do
  mkdir -p "logs/model/fine/${sc}/crop${cr}/kernel${kr}/depth${dp}/width${wd}/"
  python -u src/train.py  --gpu $gpu \
  --fine_classes --superclass $sc \
  --crop_size $cr --kernel_size $kr \
  --depth $dp --width_factor $wd  \
  | tee "logs/model/fine/${sc}/crop${cr}/kernel${kr}/depth${dp}/width${wd}/model_fine_${sc}_crop${cr}_kernel${kr}_depth${dp}_width${wd}.log"
done