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
source "/global/common/software/desi/users/adematti/cosmodesi_environment.sh" main

# --- Directories and Parameters Setup ---
DIR_SCRIPT="$HOME/desi-y1-kp45/blinding/py/"
TRACER="LRG"
TEMPLATE="shapefit-qisoqap"
THEORY="velocileptors"
OBSERVABLE="power"
TODO="emulator sampling"
OUT_DIR="/pscratch/sd/u/uendert/real_data_results/y1_full_shape/single_blinded/"
DIR_LOG="$HOME/desi-y1-kp45/blinding/scripts/log/"

# Switch to the correct directory
cd "$DIR_SCRIPT"

# --- File Naming and Path Setup ---
BASE_NAME="fit_y1_${TRACER}_${TEMPLATE}_${THEORY}_${OBSERVABLE}_${TODO// /_}"
JOB_SCRIPT_PATH="${DIR_LOG}/${BASE_NAME}.sh"
LOG_FILE_PATH="${DIR_LOG}/${BASE_NAME}.log"

echo "Log will be saved to: $LOG_FILE_PATH"

# --- Job Script Generation ---
cat > "$JOB_SCRIPT_PATH" << EOF
#!/bin/bash
srun -N 1 -n 64 -C cpu -t 04:00:00 --qos interactive --account desi -u python fit_y1.py \
    --config_path config.yaml \
    --tracer "$TRACER" \
    --template "$TEMPLATE" \
    --theory "$THEORY" \
    --observable "$OBSERVABLE" \
    --todo "$TODO" \
    --outdir "$OUT_DIR" > "$LOG_FILE_PATH" 2>&1
EOF

# --- Job Submission ---
bash "$JOB_SCRIPT_PATH"
echo "Job submitted with script: $JOB_SCRIPT_PATH"
