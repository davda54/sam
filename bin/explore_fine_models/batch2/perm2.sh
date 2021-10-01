#!/bin/sh

cr=16
kr=4
dp=22
wd=8
super_classes=("people" "vehicles_1" "flowers")


for sc in "${super_classes[@]}"; do
  mkdir -p logs/model/fine/${sc}/crop${cr}/kernel${kr}/depth${dp}/width${wd}/
  python -u src/train.py  \
  --fine_classes --superclass $sc \
  --crop_size $cr --kernel_size $kr \
  --depth $dp --width_factor $wd  \
  | tee "logs/model/fine/${sc}/crop${cr}/kernel${kr}/depth${dp}/width${wd}/model_fine_${sc}_crop${cr}_kernel${kr}_depth${dp}_width${wd}.log"
done