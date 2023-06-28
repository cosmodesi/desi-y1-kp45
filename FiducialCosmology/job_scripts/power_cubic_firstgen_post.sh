#!/bin/bash
#SBATCH -J pk_cubic_post
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --array=0-11
#SBATCH -t 01:00:00
#SBATCH -o ./out_files/%x_%j.out
#SBATCH -e ./out_files/%x_%j.err
#SBATCH -C cpu
#SBATCH -L SCRATCH
#SBATCH -A desi

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

i=${SLURM_ARRAY_TASK_ID}
tracers=(lrg elg qso)
n=$((${i}/4))
c=$((${i}%4))
tracer=${tracers[n]}

torun="/global/homes/a/alexpzfz/alexpzfz/FiducialCosmology/power_spectrum_new.py"

for ph in {0..24};
  do
  srun -n 8 python $torun cubicbox $tracer firstgen $ph -r recsym -ct 000 -cg 00${c}
  done
