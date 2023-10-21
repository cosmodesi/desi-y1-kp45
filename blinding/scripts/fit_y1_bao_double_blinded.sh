#!/bin/bash

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH --output=JOB_OUT_%x_%j.txt
#SBATCH --error=JOB_ERR_%x_%j.txt
#first steps, get environment

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

dir_script=$HOME/desi-y1-kp45/blinding/py/ 
cd $dir_script # go to the directory where the script is

tracer='LRG'
template='bao-qisoqap'
theory='dampedbao'
observable='corr'
todo='profiling'
outdir='/pscratch/sd/u/uendert/test_y1_bao/double_blinded/'
only_now_='only_now'


echo 'Running the BAO fitting pipeline'
echo $HOME/desi-y1-kp45/blinding/scripts/double_blinded/fit_y1_${tracer}_${template}_${theory}_${observable}_${todo// /_}.log

# srun -N 1 -n 4 python fit_y1.py --tracer $tracer --template $template --theory $theory --observable $observable --todo ${todo} --outdir $outdir > $HOME/desi-y1-kp45/blinding/scripts/double_blinded/fit_y1_${tracer}_${template}_${theory}_${observable}_${todo// /_}.log 2>&1
srun -N 1 -n 4 -C cpu -t 04:00:00 --qos interactive --account desi -u python fit_y1.py --tracer $tracer --template $template --theory $theory --observable $observable --todo ${todo} --outdir $outdir --double_blind y > $HOME/desi-y1-kp45/blinding/scripts/double_blinded/fit_y1_${tracer}_${template}_${theory}_${observable}_${todo// /_}.log 2>&1
echo 'Done'

# srun -N 1 -n 64 samplig
# srun -N 1 -n 4 profiling