#!/bin/bash
#SBATCH --job-name=hazard_q3_multiseed
#SBATCH --output=logs/job_%x_%A_%a.out
#SBATCH --error=logs/job_%x_%A_%a.err
#SBATCH --partition=unkillable
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem=32Gb
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

set -euo pipefail

CONFIG_FILE=${1:-configs/hazard_eval_spatial_weather_firesize_q3.yaml}
EVAL_ARGS=${EVAL_ARGS:-}
RUN_ID=${SLURM_ARRAY_TASK_ID:-0}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "Running hazard evaluation with config=${CONFIG_FILE}, run_id=${RUN_ID}"
EVAL_ARG_ARRAY=()
while IFS= read -r -d '' arg; do
    EVAL_ARG_ARRAY+=("$arg")
done < <(python - "$EVAL_ARGS" <<'PY'
import shlex
import sys

for arg in shlex.split(sys.argv[1]):
    sys.stdout.write(arg + "\0")
PY
)

python -m src.evaluate_hazard \
    --config="$CONFIG_FILE" \
    --run_id="$RUN_ID" \
    "${EVAL_ARG_ARRAY[@]}"
