#!/bin/sh

gpu=0
cr=24
kr=6
dp=16
wd=2

mkdir -p "logs/model/coarse/all/crop${cr}/kernel${kr}/depth${dp}/width${wd}/"
python -u src/train.py --gpu $gpu \
  --coarse_classes \
  --crop_size $cr --kernel_size $kr \
  --depth $dp --width_factor $wd |
  tee "logs/model/coarse/all/crop${cr}/kernel${kr}/depth${dp}/width${wd}/model_coarse_all_crop${cr}_kernel${kr}_depth${dp}_width${wd}.log"
