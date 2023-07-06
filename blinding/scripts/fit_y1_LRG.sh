#!/bin/bash

#first steps, get environment

source /global/cfs/cdirs/desi/users/adematti/cosmodesi_environment.sh main

dir_script=$HOME/desi-y1-kp45/blinding/py/ 
cd $dir_script # go to the directory where the script is

tracer='LRG'
template='bao'
theory='dampedbao'
observable='power'
todo='profiling'

echo 'Running the RSD fitting pipeline'

python fit_y1.py --tracer $tracer --template $template --theory $theory --observable $observable --todo ${todo} > $HOME/desi-y1-kp45/blinding/scripts/fit_y1_${tracer}_${template}_${theory}_${observable}_${todo// /_}.log 2>&1

echo 'Done'

# srun -N 1 -n 64 samplig
# srun -N 1 -n 4 profiling