#!/bin/bash 
#$PBS -S /bin/bash 

#PBS -N mockbox_000 
#PBS -r n 
#PBS -e logs/mockbox_000.err.$PBS_JOBID 
#PBS -o logs/mockbox_000.log.$PBS_JOBID 
#PBS -q long 
#PBS -l nodes=1:ppn=28 

module load python3 

cd $PBS_O_WORKDIR 

source activate desimock 

rm -f /dev/shm/* 

./mock_challenge_box.py -i 'AbacusSummit_base_c000_ph000/LRG_snap20_ph000.gcat.' -o 'results/results_mockbox_000' -n 20 -c 28 -z 0.8 > logs/mockbox_000.out.$PBS_JOBID 

