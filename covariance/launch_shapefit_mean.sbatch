#!/bin/bash
#SBATCH --job-name=shapemean
#SBATCH --output=shapemean.out
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular


for prior in flat #fixed
do
    srun -n64 -c4 python src/desilike_ez_v_tc_shapefit.py -cov_type analytic  -conv pre -fit_mean -prior_type $prior
    srun -n64 -c4 python src/desilike_ez_v_tc_shapefit.py -cov_type mock  -conv pre -fit_mean -prior_type $prior
done