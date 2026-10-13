#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=name@mila.quebec
#SBATCH --job-name=eval_model
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=16Gb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:TERM@300

set -euo pipefail

# --requeue lets SLURM resubmit this job (same job ID) if it is preempted or hits
# the time limit. --signal=B:TERM@300 asks SLURM to send SIGTERM 5 minutes before
# the time limit so training can shut down cleanly. src.train/Trainer already
# resumes automatically from save_dir/last.pth on restart, so no extra flags are
# needed here; just make sure save_dir points to persistent (non-tmpdir) storage.
echo "Job has been requeued/restarted ${SLURM_RESTART_COUNT:-0} time(s)."

# Config file is required; there is no universal default since it must match the
# model/data being evaluated (e.g. configs/multi_output_spatial_weather.yaml).
CONFIG_FILE=${1:?Usage: sbatch eval.sh <config_file.yaml>}
# Example: TRAIN_ARGS="--no_log_test_predicted_hexels"
TRAIN_ARGS=${TRAIN_ARGS:-}
EVAL_ARGS=${EVAL_ARGS:-}
RUN_HEXEL_EVAL=${RUN_HEXEL_EVAL:-1}

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
read -r -a EVAL_ARG_ARRAY <<< "$TRAIN_ARGS"
python -m src.evaluate_hexels \
    --config="$CONFIG_FILE" \
    "${EVAL_ARG_ARRAY[@]}"
