#!/bin/sh
conda activate sam
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --yes
conda install numpy scikit-learn pandas tqdm isort black --yes
pip install ptflops
#conda install -c conda-forge gputil --yes
