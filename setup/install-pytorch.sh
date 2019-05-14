#!/usr/bin/env bash

echo "Warning: Assumes that conda environment activated!"
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mllab-venv

python -m ipykernel install --user --name mllab-venv --display-name "Python (mllab-venv)" # TODO: Move e.g. to Makefile

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    if [ -x "$(command -v nvidia-smi)" ]; then # Check if nvidia card exists, probably not ideal
        echo "Installing cuda-supported pytorch"
        conda install -y pytorch torchvision cudatoolkit=9.0 -c pytorch
    else
        echo "Installing cpu-only pytorch"
        conda install -y pytorch-cpu torchvision-cpu -c pytorch
    fi
elif [[  "$OSTYPE" == "darwin"* ]]; then
    conda install -y pytorch torchvision -c pytorch # Only CPU version compatible via conda for mac
else
    echo "Platform not supported! Project only tested on Linux (Ubuntu) and Mac."
    exit 1
fi
