#!/bin/bash -l
#SBATCH -J MakePreal
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH -o MakePreal.out
#SBATCH -e MakePreal.err
#SBATCH -p debug
#SBATCH -C haswell
#SBATCH -A desi
#

date
#
module load python
conda activate nersc_env
#
export PYTHONPATH=${PYTHONPATH}:/global/cscratch1/sd/mmaus/ShapeFit/emulator
export PYTHONPATH=${PYTHONPATH}:/global/homes/m/mmaus/Python/velocileptors
export PYTHONPATH=${PYTHONPATH}:/global/cscratch1/sd/mmaus/Cobaya/Packages/code
export OMP_NUM_THREADS=2
#
echo "Setup done.  Starting to run code ..."
#
# Now run the code -- we do Preal and Phalofit here.
#
srun -n 32 -c 2 --unbuffered python make_emulator_pells.py  $PWD 0.59 0.31

#
date
#
