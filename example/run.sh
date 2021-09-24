#!/bin/bash

python -u train.py --fine_labels False --epochs 5 | tee output/log_fineFalse_epochs5.txt
