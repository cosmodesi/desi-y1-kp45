#!/bin/bash
#SBATCH -A desi
#SBATCH -N 32
#SBATCH -t 24:00:00
#SBATCH -C haswell
#SBATCH -q regular 

conda activate mockchallenge

srun python nersc_mock_challenge.py -i "cutsky_LRG_z0.800_AbacusSummit_base_c000_ph000.fits" -o "results/newtest.npy" -n 1 -c 32 -y 0.4 -z 0.6 >logs/newtest.txt

