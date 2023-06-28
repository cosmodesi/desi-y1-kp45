#!/bin/bash
#SBATCH -J sv3_elg_rec
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --array=0-24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH -t 00:30:00
#SBATCH -o ./out_files/%x_%j.out
#SBATCH -e ./out_files/%x_%j.err
#SBATCH -C cpu
#SBATCH -L SCRATCH
#SBATCH -A desi

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

ph=${SLURM_ARRAY_TASK_ID}

python /global/homes/a/alexpzfz/alexpzfz/FiducialCosmology/reconstruction.py cubicbox elg sv3 $ph -ct 000 -cg 000 

