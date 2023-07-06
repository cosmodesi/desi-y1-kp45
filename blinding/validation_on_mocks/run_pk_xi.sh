#!/bin/bash

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH --output=JOB_OUT_%x_%j.txt
#SBATCH --error=JOB_ERR_%x_%j.txt

# Example script

#first steps, get environment
source /global/common/software/desi/desi_environment.sh master
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

PYTHONPATH=$PYTHONPATH:$HOME/LSS/py #replace $HOME with wherever your LSS directory is cloned

cd $HOME/LSS/scripts/

tracer='QSO'
survey='main'

if [ $tracer == LRG ]; then
    zmin=0.4
    zmax=1.1
fi

if [ $tracer == ELG ]; then
    zmin=0.8
    zmax=1.6
fi

if [ $tracer == QSO ]; then
    zmin=0.8
    zmax=2.1
fi

module swap pyrecon/main pyrecon/mpi


for i in 0 1 2 3 4 5 6 7
do
    
    w0=$(python /global/homes/u/uendert/desi-y1-kp45/blinding/py/w0waf_edges_maker.py --number $i --parameter 0 --type $tracer)
    wa=$(python /global/homes/u/uendert/desi-y1-kp45/blinding/py/w0waf_edges_maker.py --number $i --parameter 1 --type $tracer)
    a=1
    blinded_index=$(($i+$a))
	
    echo $w0
    echo $wa
	echo $blinded_index
	
	basedir=/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1/mock1/LSScats/

		basedir=$basedir/blinded/jmena/test_w0${w0}_wa${wa}/LSScats/blinded
		basedir_out=$SCRATCH/blinding_mocks/test_w0${w0}_wa${wa}
	
	echo $basedir

		file=$basedir_out/LSScats/blinded/pk/pk/pkpoles_${tracer}_SGC_${zmin}_${zmax}_default_lin.npy # for pk
		if test -f "$file"; then
			echo "$file already exists"
		else
			echo "$file does not exist"

			wt='default'

			echo 'Computing pk_l'

			srun -N 1 -n 256 -u python pkrun.py --tracer $tracer --basedir $basedir --outdir $basedir_out/LSScats/blinded/pk/ --survey $survey --weight_type $wt --nran 1 --region NGC --calc_win y --zlim 0.8 2.1
			srun -N 1 -n 256 -u python pkrun.py --tracer $tracer --basedir $basedir --outdir $basedir_out/LSScats/blinded/pk/ --survey $survey --weight_type $wt --nran 1 --region SGC --calc_win y --zlim 0.8 2.1
		fi

		file=$basedir_out/LSScats/blinded/xi/smu/allcounts_${tracer}_SGC_${zmin}_${zmax}_default_lin_njack0_nran1_split20.npy # for xi
		if test -f "$file"; then
			echo "$file already exists"
		else
			echo "$file does not exist"

			wt='default'

			echo 'Computing xi_l'

			srun -N 1 -n 1 -u python xirunpc.py --tracer $tracer --basedir $basedir --outdir $basedir_out/LSScats/blinded/xi/ --survey $survey --weight_type $wt --corr_type smu --nran 1 --nthreads 256 --njack 0 --region NGC --zlim 0.8 2.1
			srun -N 1 -n 1 -u python xirunpc.py --tracer $tracer --basedir $basedir --outdir $basedir_out/LSScats/blinded/xi/ --survey $survey --weight_type $wt --corr_type smu --nran 1 --nthreads 256 --njack 0 --region SGC --zlim 0.8 2.1
		fi	
		unset basedir
    
done