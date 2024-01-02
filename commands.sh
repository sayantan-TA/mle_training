#!/bin/bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# Activate the Conda environment in the current shell
source ~/miniconda3/bin/activate

conda init bash
conda init zsh

# Create and activate the Conda environment
conda env create -n mle-dev -f deploy/conda/env.yml
conda activate mle-dev
pip install .
bash
