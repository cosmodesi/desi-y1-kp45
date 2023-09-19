#!/bin/bash
#SBATCH --job-name=bao2ximean
#SBATCH --output=bao2ximean.out
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular


for prior in fixed gauss flat
do
    srun -n64 -c4 python src/desilike_ez_v_tc_bao2.py -cov_type analytic  -conv sym -fit_mean -prior_type $prior -rerun -rescaled
    srun -n64 -c4 python src/desilike_ez_v_rc_bao2.py -cov_type mock  -conv sym -fit_mean -prior_type $prior -rerun
    
done