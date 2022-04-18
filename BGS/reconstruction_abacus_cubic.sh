#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=haswell
#SBATCH -q debug
#SBATCH -t 00:30:00

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module load openmpi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=64

time srun -n 1 python $HOME/data/mock_challenge/bgs/reconstruction_abacus_cubic.py
