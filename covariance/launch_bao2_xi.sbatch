#!/bin/bash
#SBATCH --job-name=bao2xi
#SBATCH --output=bao2xi.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=2:00:00
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular


module load parallel
#seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_rc_bao2.py -cov_type mock -nchunks 128 -chunk {} -conv sym -rerun
#seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_rc_bao2.py -cov_type analytic -nchunks 128 -chunk {} -conv sym -rescaled -rerun


seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_rc_bao2.py -cov_type mock -nchunks 128 -chunk {} -conv sym  -chain_input data/desilike_mock_sym_bao2_minuit_xi//chain_mean_bao2_gauss_0.npy -seed 0 
seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_rc_bao2.py -cov_type analytic -nchunks 128 -chunk {} -conv sym -rescaled  -chain_input data/desilike_mock_sym_bao2_minuit_xi//chain_mean_bao2_gauss_0.npy -seed 0

#seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_tc_bao2.py -cov_type mock -nchunks 128 -chunk {} -conv sym -rerun
#seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_tc_bao2.py -cov_type analytic -nchunks 128 -chunk {} -conv sym -rescaled -rerun


seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_tc_bao2.py -cov_type mock -nchunks 128 -chunk {} -conv sym  -chain_input data/desilike_mock_sym_bao2_minuit_pk//chain_mean_bao2_gauss_0.npy
seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_tc_bao2.py -cov_type analytic -nchunks 128 -chunk {} -conv sym -rescaled  -chain_input data/desilike_mock_sym_bao2_minuit_pk//chain_mean_bao2_gauss_0.npy