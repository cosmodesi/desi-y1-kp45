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

# --- Environment Setup ---
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# --- Directories and Parameters Setup ---
DIR_SCRIPT="$HOME/desi-y1-kp45/blinding/py"
TRACER="LRG"
TEMPLATE="shapefit-qisoqap"
THEORY="velocileptors"
OBSERVABLE="corr"
TODO="sampling"
OUT_DIR="/pscratch/sd/u/uendert/mocks_restuls/y1_full_shape/unblinded"
DIR_LOG="$HOME/desi-y1-kp45/blinding/scripts/log"

# Switch to the appropriate directory
cd "$DIR_SCRIPT"

echo 'Running the RSD fitting pipeline'

# --- Log File Setup ---
LOG_FILE="${DIR_LOG}/fit_y1_${TRACER}_${TEMPLATE}_${THEORY}_${OBSERVABLE}_${TODO// /_}_mocks_unblinded.log"
echo "Log will be saved to: $LOG_FILE"

# --- Job Execution ---
srun -N 1 -n 64 -C cpu -t 04:00:00 \
     --qos interactive \
     --account desi \
     -u python fit_y1.py \
     --config_path config_mocks_unblinded.yaml \
     --tracer $TRACER \
     --template $TEMPLATE \
     --theory $THEORY \
     --observable $OBSERVABLE \
     --todo $TODO \
     --outdir $OUT_DIR \
     --zlim 0.8 1.1 > "$LOG_FILE" 2>&1

echo 'Done'

