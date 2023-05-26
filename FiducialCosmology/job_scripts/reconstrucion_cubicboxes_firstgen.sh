#!/bin/bash
#SBATCH -J rec_cubic
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --array=0-14
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
j=$((${id}/5))
c=$((${id}%5))
tracer=${tracers[j]}

torun="/global/homes/a/alexpzfz/FiducialCosmology/reconstruction.py"

for ph in {0..24};
  do
  python $torun cubicbox $tracer firstgen 000 00$c $ph
done
