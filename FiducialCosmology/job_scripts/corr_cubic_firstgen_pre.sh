#!/bin/bash
#SBATCH -J xi_cubic_pre
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --array=0-2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH -t 01:00:00
#SBATCH -o ./out_files/%x_%j.out
#SBATCH -e ./out_files/%x_%j.err
#SBATCH -C cpu
#SBATCH -L SCRATCH
#SBATCH -A desi

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

tracers=(lrg elg qso)
id=${SLURM_ARRAY_TASK_ID}
tracer=${tracers[id]}

torun="/global/homes/a/alexpzfz/alexpzfz/FiducialCosmology/correlation_function.py"

for c in {0..4};
do
for ph in {0..24};
  do
  srun -n 2 python $torun cubicbox $tracer firstgen $ph -ct 000 -cg 00${c}
  done
done
