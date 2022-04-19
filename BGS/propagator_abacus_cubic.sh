#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --constraint=haswell
#SBATCH -q shared
#SBATCH -t 2:00:00

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module load openmpi

srun -n 32 python $HOME/data/mock_challenge/bgs/propagator.py
