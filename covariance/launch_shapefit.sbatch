#!/bin/bash
#SBATCH --job-name=shapefit
#SBATCH --output=shapefit.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=2:00:00
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular


module load parallel
seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_tc_shapefit.py -cov_type mock -nchunks 128 -chunk {} -conv pre -rerun
seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_tc_shapefit.py -cov_type analytic -nchunks 128 -chunk {} -conv pre -rerun