# Project 4: Large-scale graph mining - ML-Lab Summer 2019


### Installation
If you have not yet installed conda, just run `make`, make sure *.sh are executable (i.e. chmod +x ...). The script will first try to sinstall anaconda for your platform 
(only MacOS and Linux supported at the moment) and then creates the conda environment and installs necessary dependencies.  
If you already have anaconda installed, just run `make environment`.
Please note that it will try to detect if you have cuda installed and downloads the cpu-only version if not.

### Repository Structure
As our work was mostly separated into two parts, you can find further instructions on 
how to re-run our experiments in the respective folders, `nri` and `graph`.

#### Run on GPU
To run a program on a specific GPU only, execute
`CUDA_VISIBLE_DEVICES=2 python train.py` or `CUDA_VISIBLE_DEVICES=2,3 jupyter notebook`


### Useful Commands
#### TMUX
Start new session: `tmux new -s session-name`  
Detach from session: `ctrl-b d`   
Show active sessions: `tmux ls`  
Attach to running: `tmux attach -t session-name`  



### Contribution
I'd propose to keep the overall structure similar to the one 
in [this Pytorch Template](https://github.com/victoresque/pytorch-template). This is optional of course though. :)
Right now, the logger package is directly used in our project.