#!/bin/bash

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH --output=JOB_OUT_%x_%j.txt
#SBATCH --error=JOB_ERR_%x_%j.txt


# Example script

#first steps, get environment

source /global/cfs/cdirs/desi/users/adematti/cosmodesi_environment.sh main

# # The following two lines may not be needed
# export CXI_FORK_SAFE=1
# export CXI_FORK_SAFE_HP=1

tracer='QSO'
survey='Y1'
region='NGC'



basedir_out=/global/cfs/cdirs/desi/users/uendert/profiling/on_the_fly_GaussianCovariance_matrix/fs/

echo 'Running the RSD fitting pipeline'

srun -N 1 -n 4 python py/bao_fs_fit.py --type $tracer --survey $survey --basedir_out $basedir_out --verspec iron --version v0.1 --region $region --todo emulator fs profiling

#> ${basedir_out}/bao_fit_pk_blinded.log 2>&1

