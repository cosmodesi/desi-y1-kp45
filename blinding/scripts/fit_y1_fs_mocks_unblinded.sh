#!/bin/bash

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH --output=JOB_OUT_%x_%j.txt
#SBATCH --error=JOB_ERR_%x_%j.txt

# First steps, set up the environment.
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

dir_script=$HOME/desi-y1-kp45/blinding/py/
cd $dir_script # Go to the directory where the script is.

tracer='LRG'
template='shapefit-qisoqap'
theory='velocileptors'
observable='corr'
todo='sampling'
outdir='/pscratch/sd/u/uendert/mocks_restuls/y1_full_shape/unblinded/'

echo 'Running the RSD fitting pipeline'


# Log file path
log_file=$HOME/desi-y1-kp45/blinding/scripts/log/fit_y1_${tracer}_${template}_${theory}_${observable}_${todo// /_}_mocks_unblinded.log
echo $log_file

# Execute the python script
srun -N 1 -n 64 -C cpu -t 04:00:00 --qos interactive --account desi -u python fit_y1.py --config_path config_mocks_unblinded.yaml --tracer $tracer --template $template --theory $theory --observable $observable --todo ${todo} --outdir $outdir$i --zlim 0.8 1.1  > $log_file 2>&1


echo 'Done'
