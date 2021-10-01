#!/bin/sh

gpu=7
cr=32
kr=8
dp=28
wd=6
super_classes=("food_containers" "large_man-made_outdoor_things" "non-insect_invertebrates")

for sc in "${super_classes[@]}"; do
  mkdir -p logs/model/fine/${sc}/crop${cr}/kernel${kr}/depth${dp}/width${wd}/
  python -u src/train.py --gpu $gpu \
  --fine_classes --superclass $sc \
  --crop_size $cr --kernel_size $kr \
  --depth $dp --width_factor $wd  \
  | tee logs/model/fine/${sc}/crop${cr}/kernel${kr}/depth${dp}/width${wd}/ \
  model_fine_${sc}_crop${cr}_kernel${kr}_depth${dp}_width${wd}.log
done