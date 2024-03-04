#!/bin/bash
#SBATCH --job-name=baoxi
#SBATCH --output=baoxi.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=2:00:00
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular


module load parallel
seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_rc_bao.py -cov_type mock -nchunks 128 -chunk {} -conv sym -seed 1
seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_rc_bao.py -cov_type analytic -nchunks 128 -chunk {} -conv sym -rescaled -seed 1


#seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_rc_bao2.py -cov_type mock -nchunks 128 -chunk {} -conv sym -rerun -chain_input data/desilike_mock_sym_bao2_minuit_xi//chain_mean_bao2_gauss_0.npy
#seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_rc_bao2.py -cov_type analytic -nchunks 128 -chunk {} -conv sym -rescaled -rerun -chain_input data/desilike_mock_sym_bao2_minuit_xi//chain_mean_bao2_gauss_0.npy

seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_tc_bao.py -cov_type mock -nchunks 128 -chunk {} -conv sym -seed 1
seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_tc_bao.py -cov_type analytic -nchunks 128 -chunk {} -conv sym -rescaled -seed 1


#seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_tc_bao2.py -cov_type mock -nchunks 128 -chunk {} -conv sym -rerun -chain_input data/desilike_mock_sym_bao2_minuit_pk//chain_mean_bao2_gauss_0.npy
#seq 0 1 127 | parallel -I'{}' python src/desilike_ez_v_tc_bao2.py -cov_type analytic -nchunks 128 -chunk {} -conv sym -rescaled -rerun -chain_input data/desilike_mock_sym_bao2_minuit_pk//chain_mean_bao2_gauss_0.npy