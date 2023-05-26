#!/bin/bash
#SBATCH -J elg_power
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --array=0-4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -C cpu
#SBATCH -L SCRATCH
#SBATCH -A desi

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

export FID=${SLURM_ARRAY_TASK_ID}

for CAT in 003 004;
do
  srun -n 8 python /global/homes/a/alexpzfz/FiducialCosmology/power_spectrum_CubicBox_Antoine.py $CAT 00$FID
done
