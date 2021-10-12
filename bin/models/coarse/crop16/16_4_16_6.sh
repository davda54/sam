#!/bin/sh

gpu=1
cr=16
kr=4
dp=16
wd=6

mkdir -p "logs/model/coarse/${sc}/crop${cr}/kernel${kr}/depth${dp}/width${wd}/"
python -u src/train.py --gpu $gpu \
  --coarse_classes \
  --crop_size $cr --kernel_size $kr \
  --depth $dp --width_factor $wd |
  tee "logs/model/coarse/${sc}/crop${cr}/kernel${kr}/depth${dp}/width${wd}/model_coarse_${sc}_crop${cr}_kernel${kr}_depth${dp}_width${wd}.log"
