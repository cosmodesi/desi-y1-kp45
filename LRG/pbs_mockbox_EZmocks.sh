#!/bin/bash 
#$PBS -S /bin/bash 

#PBS -N mockbox_EZmocks 
#PBS -r n 
#PBS -e logs/mockbox_EZmocks.err.$PBS_JOBID 
#PBS -o logs/mockbox_EZmocks.log.$PBS_JOBID 
#PBS -q long 
#PBS -l nodes=1:ppn=48 

module load python3 

cd $PBS_O_WORKDIR 

source activate desimock 

rm -f /dev/shm/* 

./mock_challenge_box_EZmocks.py -n 20 -c 48 -z 0.8 > logs/mockbox_EZmocks.out.$PBS_JOBID 

