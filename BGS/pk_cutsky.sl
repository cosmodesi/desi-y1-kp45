#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
##SBATCH --constraint=haswell
#SBATCH --constraint=cpu
#SBATCH -q debug
##SBATCH -q regular 
#SBATCH -t 30:00
#SBATCH -J pk_cutsky
#SBATCH -o ./stdout/%x.o%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zd585612@ohio.edu

# do it in the login node
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
#module load openmpi


#export MPI4PY_RC_RECV_MPROBE='False'   # add it for the perlmutter
export HDF5_USE_FILE_LOCKING=FALSE

export OMP_NUM_THREADS=8

time srun -n 32 --cpu-bind=cores python bgs_cutsky_power.py

