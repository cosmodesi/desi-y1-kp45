#!/bin/bash -l
#SBATCH -J Fit_PS
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -o shapefit.out
#SBATCH -e shapefit.err
#SBATCH -p debug
#SBATCH -C haswell
#SBATCH -A desi

date
#
module load python
conda activate nersc_env
#

export PYTHONPATH=${PYTHONPATH}:/global/cscratch1/sd/mmaus/ShapeFit
export PYTHONPATH=${PYTHONPATH}:/global/cscratch1/sd/mmaus/ShapeFit/rsd_likelihood
# export PYTHONPATH=${PYTHONPATH}:/global/cscratch1/sd/mmaus/new_template/Cobaya_template/emulator/template/emu

echo "Setup done.  Starting to run code ..."

srun -n 8 -c 8 --unbuffered cobaya-run shapefit_z1_pk.yaml