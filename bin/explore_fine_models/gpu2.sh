#!/bin/sh

gpu=2
cr=16
kr=4
dp=28
wd=10
super_classes=("food_containers" "large_man-made_outdoor_things" "non-insect_invertebrates")

for sc in "${super_classes[@]}"; do
  python -u src/train.py --gpu $gpu \
  --fine_classes --superclass $sc \
  --crop_size $cr --kernel_size $kr \
  --depth $dp --width_factor $wd  \
  | tee logs/mode/fine/${sc}/crop${cr}/kernel${kr}/depth${dp}/width${wd}/ \
  model_fine_${sc}_crop${cr}_kernel${kr}_depth${dp}_width${wd}.log
done