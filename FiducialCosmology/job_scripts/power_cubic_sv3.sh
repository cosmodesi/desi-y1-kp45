#!/bin/bash
#SBATCH -J pk_cubic_sv3
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --array=0-11
#SBATCH -t 00:10:00
#SBATCH -o ./out_files/%x_%j.out
#SBATCH -e ./out_files/%x_%j.err
#SBATCH -C cpu
#SBATCH -L SCRATCH
#SBATCH -A desi

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

j=${SLURM_ARRAY_TASK_ID}
cosmo=(003 004)
i=$((${j}/6))
c=${cosmo[i]}
ph=$((${j}%6))

torun="/global/homes/a/alexpzfz/alexpzfz/FiducialCosmology/power_spectrum.py"

srun -n 2 python $torun cubicbox elg sv3 $ph -ct $c -cg $c
