#!/bin/bash

# Slurm job directives
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH --output=JOB_OUT_%x_%j.txt
#SBATCH --error=JOB_ERR_%x_%j.txt

# Set environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# Directories and parameters
DIR_SCRIPT="$HOME/desi-y1-kp45/blinding/py/"
TRACER='LRG'
TEMPLATE='shapefit-qisoqap'
THEORY='velocileptors'
OBSERVABLE='corr'
TODO='emulator sampling'
OUTDIR='/pscratch/sd/u/uendert/mocks_restuls/y1_full_shape/'

# Switch to script directory
cd "$DIR_SCRIPT"

echo 'Running the RSD fitting pipeline'

# Define test cases
TEST_CASES=(
  "test_w0-0.9040043101843285_wa0.025634205416364297_fnl20"
  "test_w0-0.9057030601797708_wa-0.6831142329608426_fnl20"
  "test_w0-0.970439944958287_wa-0.507777992481059_fnl20"
  "test_w0-0.996229742129104_wa0.28930866494014884_fnl20"
  "test_w0-1.0485430984101343_wa0.14015686872763022_fnl20"
  "test_w0-1.106392086529483_wa0.45478607672455995_fnl20"
  "test_w0-1.1616966626392298_wa0.3746115553255438_fnl20"
  "test_w0-1.233469858595847_wa0.7658531629974685_fnl20"
)

# Process each test case
for TEST in "${TEST_CASES[@]}"; do
    # Update the config file for the current test case
    sed "s|{test_case}|$TEST|g" config_mocks.yaml > config_mocks_temp.yaml

    # Define log file
    DIR_LOG="$HOME/desi-y1-kp45/blinding/scripts/log/"
    LOG_FILE="${DIR_LOG}fit_y1_${TRACER}_${TEMPLATE}_${THEORY}_${OBSERVABLE}_${TODO// /_}.log"
    
    echo "Logging to: $LOG_FILE"

    # Execute the fitting pipeline
    srun -N 1 -n 64 -C cpu -t 04:00:00 --qos interactive --account desi -u python fit_y1.py \
     --config_path config_mocks_temp.yaml \
     --tracer "$TRACER" --template "$TEMPLATE" \
     --theory "$THEORY" --observable "$OBSERVABLE" \
     --todo "$TODO" --outdir "${OUTDIR}${TEST}" > "$LOG_FILE" 2>&1
done

# Clean up temporary config file
rm config_mocks_temp.yaml

echo 'Process Completed'
