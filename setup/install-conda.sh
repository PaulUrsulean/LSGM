#!/usr/bin/env bash

echo Installing Anaconda

# Get correct anaconda script for platform
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    wget -O anaconda_install.sh https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
elif [[  "$OSTYPE" == "darwin"* ]]; then
    wget -O anaconda_install.sh https://repo.anaconda.com/archive/Anaconda3-2019.03-MacOSX-x86_64.sh
else
    echo "Platform not supported! Project only tested on Linux (Ubuntu) and Mac."
    exit 1
fi

# Install anaconda
bash anaconda_install.sh -b -p $HOME/anaconda3/
rm anaconda_install.sh
eval "$($HOME/anaconda3/bin/conda shell.$(basename $SHELL) hook)"
source ~/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc

conda init
conda update --yes conda