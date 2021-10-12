#!/bin/sh
conda create -n sam python=3.9.7
conda init bash
conda activate sam
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --yes
conda install numpy scikit-learn pandas tqdm isort black --yes
conda install -c conda-forge gputil --yes
pip install ptflops
