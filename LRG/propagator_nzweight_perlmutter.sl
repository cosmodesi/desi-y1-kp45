#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --constraint=cpu
##SBATCH -q debug
#SBATCH -q regular
#SBATCH -t 2:30:00
#SBATCH -J prop_lrg_cutsky

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

#cap=sgc
cap_list=(ngc)
#cap_list=(ngc sgc)

#0.4<z<0.6, 0.6<z<0.8 or 0.8<z<1.1
zmin=0.4
zmax=0.6

#zmin=0.6
#zmax=0.8

#zmin=0.8
#zmax=1.1

recon_nmesh=1024

add_nzweight=True
#add_nzweight=False

if [ ${add_nzweight} = "True" ]; then
    dir_nzweight="with_nzweight"
else
    dir_nzweight="no_nzweight"
fi

input_ic_dir="/global/cfs/cdirs/desi/users/jerryou/MockChallenge/y1_mockchallenge/reconstruction/AbacusSummit_base_c000_ph000/cutsky/IC/new_IC/" 

##input_tracer_dir="/pscratch/sd/j/jerryou/y1_mockchallenge/reconstruction/cutsky/LRG/set_nmesh/recon_catalogues/"
# --- for the output from nz weight included in the IC weight
output_dir="/pscratch/sd/j/jerryou/y1_mockchallenge/propagator/cutsky/LRG/new_IC/${dir_nzweight}/set_nmesh/"

#nmesh_list=(512)
nmesh_list=(1024)        ## set mesh size for propagator calculation, nmesh=1024 fails for NGC ran in Cori due to out-of-memoery

for cap in ${cap_list[*]}; do
  CAP=$(echo $cap | tr '[:lower:]' '[:upper:]')
  echo $CAP

  input_tracer_dir="/pscratch/sd/j/jerryou/y1_mockchallenge/reconstruction/cutsky/LRG/set_nmesh/${CAP}/recon_catalogues/"

  for nmesh in ${nmesh_list[*]}; do
    echo "nmesh=$nmesh"
    time srun -n 16 --cpu-bind=cores python lrg_cutsky_propagator.py --cap $cap --input_ic_dir ${input_ic_dir} --input_tracer_dir ${input_tracer_dir} --output_dir ${output_dir} --nmesh $nmesh --zmin $zmin --zmax $zmax --recon_nmesh ${recon_nmesh} --add_nzweight ${add_nzweight}
  done
done
