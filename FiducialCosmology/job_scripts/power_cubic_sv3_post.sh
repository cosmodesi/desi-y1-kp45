#!/bin/bash
#SBATCH -J pk_cubic_sv3
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --array=0-24
#SBATCH -t 00:30:00
#SBATCH -o ./out_files/%x_%j.out
#SBATCH -e ./out_files/%x_%j.err
#SBATCH -C cpu
#SBATCH -L SCRATCH
#SBATCH -A desi

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main


ph=${SLURM_ARRAY_TASK_ID}

torun="/global/homes/a/alexpzfz/alexpzfz/FiducialCosmology/power_spectrum.py"

srun -n 4 python $torun cubicbox elg sv3 $ph -r recsym -ct 000 -cg 000
