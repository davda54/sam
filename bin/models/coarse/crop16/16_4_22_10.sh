#!/bin/sh

gpu=5
cr=16
kr=4
dp=22
wd=10

mkdir -p "logs/model/coarse/all/crop${cr}/kernel${kr}/depth${dp}/width${wd}/"
python -u src/train.py --gpu $gpu \
  --coarse_classes \
  --crop_size $cr --kernel_size $kr \
  --depth $dp --width_factor $wd |
  tee "logs/model/coarse/all/crop${cr}/kernel${kr}/depth${dp}/width${wd}/model_coarse_all_crop${cr}_kernel${kr}_depth${dp}_width${wd}.log"
