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
source "/global/cfs/cdirs/desi/users/adematti/cosmodesi_environment.sh" main

# --- Directories and Parameters setup ---
DIR_SCRIPT="$HOME/desi-y1-kp45/blinding/py/"
TRACER="LRG"
TEMPLATE="bao-qisoqap"
THEORY="dampedbao"
OBSERVABLE="power"
TODO="profiling"
OUT_DIR="/pscratch/sd/u/uendert/y1_bao/"
ONLY_NOW_FLAG="only_now"
DIR_LOG="$HOME/desi-y1-kp45/blinding/scripts"
CONFIG_PATH="path_to_your_yaml_file.yaml"  # Update this with your YAML file's path

# Switch to the correct directory
cd "$DIR_SCRIPT"

# --- File naming and path setup ---
LOG_FILE_BASE_NAME="fit_y1_${TRACER}_${TEMPLATE}_${THEORY}_${OBSERVABLE}_${TODO// /_}"
LOG_FILE_WITH_ONLY_NOW="${DIR_LOG}/${LOG_FILE_BASE_NAME}_${ONLY_NOW_FLAG}.log"
LOG_FILE="${DIR_LOG}/${LOG_FILE_BASE_NAME}.log"

# --- Execute the pipeline ---
echo 'Running the BAO fitting pipeline'

# Run with only_now flag
python fit_y1.py \
    --config_path "$CONFIG_PATH" \
    --tracer $TRACER \
    --template $TEMPLATE \
    --theory $THEORY \
    --observable $OBSERVABLE \
    --todo $TODO \
    --outdir $OUT_DIR \
    --only_now > "$LOG_FILE_WITH_ONLY_NOW" 2>&1

# Run without only_now flag
python fit_y1.py \
    --config_path "$CONFIG_PATH" \
    --tracer $TRACER \
    --template $TEMPLATE \
    --theory $THEORY \
    --observable $OBSERVABLE \
    --todo $TODO \
    --outdir $OUT_DIR > "$LOG_FILE" 2>&1

echo 'Done'

# srun -N 1 -n 64 sampling
# srun -N 1 -n 4 profiling