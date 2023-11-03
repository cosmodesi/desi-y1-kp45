#!/bin/bash

# WARNING: This script needs to be updated to include YAML file

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
source "/global/common/software/desi/users/adematti/cosmodesi_environment.sh" main

# --- Directories and Parameters setup ---
DIR_SCRIPT="$HOME/desi-y1-kp45/blinding/py/"
LOG_DIR="$HOME/desi-y1-kp45/blinding/scripts/double_blinded"
OUT_DIR="/pscratch/sd/u/uendert/test_y1_bao/double_blinded/"

TRACER="LRG"
TEMPLATE="bao-qisoqap"
THEORY="dampedbao"
OBSERVABLE="corr"
TODO="profiling"
CONFIG_PATH="path_to_your_yaml_file.yaml"  # Update this with your YAML file's path

# Switch to the correct directory
cd "$DIR_SCRIPT"

# Log file name 
LOG_FILE="${LOG_DIR}/fit_y1_${TRACER}_${TEMPLATE}_${THEORY}_${OBSERVABLE}_${TODO// /_}.log"

echo "Running the BAO fitting pipeline"
echo "$LOG_FILE"

# Execute the pipeline
srun -N 1 -n 4 -C cpu -t 04:00:00 --qos interactive --account desi -u python fit_y1.py \
    --config_path "$CONFIG_PATH" \
    --tracer $TRACER \
    --template $TEMPLATE \
    --theory $THEORY \
    --observable $OBSERVABLE \
    --todo $TODO \
    --outdir $OUT_DIR > "$LOG_FILE" 2>&1

echo 'Done'

