#!/bin/bash
#SBATCH -J rec_cubic
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --array=0-4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH -t 00:40:00
#SBATCH -o ./out_files/%x_%j.out
#SBATCH -e ./out_files/%x_%j.err
#SBATCH -C cpu
#SBATCH -L SCRATCH
#SBATCH -A desi

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

c=${SLURM_ARRAY_TASK_ID}
tracer=elg

torun="/global/homes/a/alexpzfz/FiducialCosmology/reconstruction.py"

for ph in {17..24};
  do
  python $torun cubicbox $tracer firstgen 000 00$c $ph
done
