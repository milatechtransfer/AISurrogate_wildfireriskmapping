#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=<your_email>@<your_domain>
#SBATCH --job-name=unet_multitask
#SBATCH --output=logs/job_%x_%A_%a.out
#SBATCH --error=logs/job_%x_%A_%a.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=09:59:00
#SBATCH --mem=64Gb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-2                # Launches 3 parallel runs (0,1,2); override at submit time, e.g. `sbatch --array=0-4 ...` for 5 runs.
#SBATCH --requeue
#SBATCH --signal=B:TERM@300

set -euo pipefail

echo "Job has been requeued/restarted ${SLURM_RESTART_COUNT:-0} time(s)."

# Capture the first argument, default to the common pipeline config.
CONFIG_FILE=${1:-configs/bp_common_input_pipeline.yaml}
TRAIN_ARGS=${TRAIN_ARGS:-}
EVAL_ARGS=${EVAL_ARGS:-}
RUN_HEXEL_EVAL=${RUN_HEXEL_EVAL:-1}

# Map the SLURM array task ID to a run index (0 when run outside an array job).
# Forwarded to src.train/src.evaluate_hexels as --run_id so each parallel run
# gets its own derived seed, save_dir, and Comet experiment name.
RUN_ID=${SLURM_ARRAY_TASK_ID:-0}
echo "Starting run RUN_ID=${RUN_ID}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate
# Flush stdout/stderr immediately so log lines aren't lost if the job is
# preempted before Python's internal buffers would otherwise flush.
export PYTHONUNBUFFERED=1
LOGGER_ENABLED=$(python - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

import yaml

with Path(sys.argv[1]).open() as handle:
    config = yaml.safe_load(handle)

print(str(config.get("logger", {}).get("enabled", True)).lower())
PY
)

if [[ "$LOGGER_ENABLED" == "true" && -z "${COMET_API_KEY:-}" ]]; then
    echo "ERROR: COMET_API_KEY must be exported when logger.enabled=true." >&2
    exit 1
fi

echo "Using data.root_dir directly from config: $CONFIG_FILE"

echo "Running training with config: $CONFIG_FILE"
read -r -a TRAIN_ARG_ARRAY <<< "$TRAIN_ARGS"
python -m src.train \
    --config="$CONFIG_FILE" \
    --run_id="$RUN_ID" \
    "${TRAIN_ARG_ARRAY[@]}"

# if [[ "$RUN_HEXEL_EVAL" == "1" ]]; then
#     echo "Running evaluation with config: $CONFIG_FILE"
#     read -r -a EVAL_ARG_ARRAY <<< "$EVAL_ARGS"
#     python -m src.evaluate_hexels \
#         --config="$CONFIG_FILE" \
#         --run_id="$RUN_ID" \
#         "${EVAL_ARG_ARRAY[@]}"
# else
#     echo "Skipping hexel evaluation because RUN_HEXEL_EVAL=${RUN_HEXEL_EVAL}"
# fi
