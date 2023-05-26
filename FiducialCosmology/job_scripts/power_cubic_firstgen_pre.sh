#!/bin/bash
#SBATCH -J pk_cubic_pre
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --array=0-4
#SBATCH -t 03:00:00
#SBATCH -o ./out_files/%x_%j.out
#SBATCH -e ./out_files/%x_%j.err
#SBATCH -C cpu
#SBATCH -L SCRATCH
#SBATCH -A desi

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

c=${SLURM_ARRAY_TASK_ID}

torun="/global/homes/a/alexpzfz/FiducialCosmology/power_spectrum.py"

for tracer in lrg elg qso;
do
for ph in {0..24};
  do
  srun -n 4 python $torun cubicbox $tracer firstgen $ph -ct 000 -cg 00${c}
  done
done
