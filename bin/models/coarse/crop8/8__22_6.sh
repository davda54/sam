#!/bin/sh

cr=8
dp=22
wd=6
gpu=4

mkdir -p "logs/model/coarse/all/crop${cr}/kernel${kr}/depth${dp}/width${wd}/"
python -u src/train.py --gpu $gpu \
  --coarse_classes \
  --crop_size $cr\
  --depth $dp --width_factor $wd |
  tee "logs/model/coarse/all/crop${cr}/kernel${kr}/depth${dp}/width${wd}/model_coarse_all_crop${cr}_kernel${kr}_depth${dp}_width${wd}.log"
