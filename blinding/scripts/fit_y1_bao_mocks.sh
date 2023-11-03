#!/bin/bash

# --- Slurm job directives ---
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=256
#SBATCH --output=JOB_OUT_%x_%j.txt
#SBATCH --error=JOB_ERR_%x_%j.txt

# --- Environment setup ---
source "/global/cfs/cdirs/desi/users/adematti/cosmodesi_environment.sh" main

# --- Directories and Parameters setup ---
DIR_SCRIPT="$HOME/desi-y1-kp45/blinding/py/"
TRACER="LRG"
TEMPLATE="bao-qisoqap"
THEORY="dampedbao"
OBSERVABLE="corr"
TODO="profiling"
OUT_DIR_BASE="/pscratch/sd/u/uendert/mocks_restuls/y1_bao/"

# Switch to the correct directory
cd "$DIR_SCRIPT"

# --- Test Cases Array ---
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

# --- Loop through each test case ---
for TEST_CASE in "${TEST_CASES[@]}"
do
    # Update configuration for current test case and write to a temporary file.
    sed "s|{test_case}|$TEST_CASE|g" config_mocks.yaml > config_mocks_temp.yaml

    # Execute the pipeline for current test case
    echo "Running the BAO fitting pipeline for $TEST_CASE"
    python fit_y1.py \
        --config_path config_mocks_temp.yaml \
        --tracer $TRACER \
        --template $TEMPLATE \
        --theory $THEORY \
        --observable $OBSERVABLE \
        --todo $TODO \
        --outdir "${OUT_DIR_BASE}${TEST_CASE}"
done

# Cleanup temporary files
rm config_mocks_temp.yaml

echo 'Process Completed'
