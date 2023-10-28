#!/bin/bash

#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH --output=JOB_OUT_%x_%j.txt
#SBATCH --error=JOB_ERR_%x_%j.txt

# First steps, set up the environment.
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

dir_script=$HOME/desi-y1-kp45/blinding/py/
cd $dir_script # Go to the directory where the script is.

tracer='LRG'
template='shapefit-qisoqap'
theory='velocileptors'
observable='corr'
todo='emulator sampling'
outdir='/pscratch/sd/u/uendert/mocks_restuls/y1_full_shape/'

echo 'Running the RSD fitting pipeline'

# Loop through each test case.
for i in "test_w0-0.9040043101843285_wa0.025634205416364297_fnl20" \
        "test_w0-0.9057030601797708_wa-0.6831142329608426_fnl20" \
        "test_w0-0.970439944958287_wa-0.507777992481059_fnl20" \
        "test_w0-0.996229742129104_wa0.28930866494014884_fnl20" \
        "test_w0-1.0485430984101343_wa0.14015686872763022_fnl20" \
        "test_w0-1.106392086529483_wa0.45478607672455995_fnl20" \
        "test_w0-1.1616966626392298_wa0.3746115553255438_fnl20" \
        "test_w0-1.233469858595847_wa0.7658531629974685_fnl20"
do
    # Use the sed command to replace the placeholder and write to a temporary file.
    sed "s|{test_case}|$i|g" config_mocks.yaml > config_mocks_temp.yaml

    # Log file path
    log_file=$HOME/desi-y1-kp45/blinding/scripts/log/fit_y1_${tracer}_${template}_${theory}_${observable}_${todo// /_}.log
    echo $log_file

    # Execute the python script
    srun -N 1 -n 64 -C cpu -t 04:00:00 --qos interactive --account desi -u python fit_y1.py --config_path config_mocks_temp.yaml --tracer $tracer --template $template --theory $theory --observable $observable --todo ${todo} --outdir $outdir$i > $log_file 2>&1
done

# Remove the temporary file after the loop ends
rm config_mocks_temp.yaml

echo 'Done'
