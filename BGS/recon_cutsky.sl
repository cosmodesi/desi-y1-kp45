#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
##SBATCH --constraint=haswell
#SBATCH -C cpu
#SBATCH -q debug
##SBATCH -q regular
#SBATCH -t 30:00
#SBATCH -J recon_cutsky
#SBATCH -o ./stdout/%x.o%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zd585612@ohio.edu

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
##module load openmpi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=64

time srun -n 1 python bgs_cutsky_recon.py
