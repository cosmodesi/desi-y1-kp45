#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --constraint=haswell
#SBATCH -q debug
#SBATCH -t 00:30:00

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module load openmpi
export HDF5_USE_FILE_LOCKING=FALSE

time srun -n 32 python $HOME/data/mock_challenge/bgs/pk_cubic.py
