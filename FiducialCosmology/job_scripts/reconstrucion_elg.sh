#!/bin/bash
#SBATCH -J elg_rec
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --array=1-5%10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH -t 01:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -C cpu
#SBATCH -L SCRATCH
#SBATCH -A desi

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

export FID=`expr ${SLURM_ARRAY_TASK_ID} - 1`

for CAT in 003 004;
do
  python /global/homes/a/alexpzfz/FiducialCosmology/reconstruction_CubicBox_Antoine.py $CAT 00$FID
done
