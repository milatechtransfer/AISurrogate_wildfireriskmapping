#!/bin/bash
#SBATCH --job-name=eval_model_multiseed
#SBATCH --output=logs/job_%x_%A_%a.out
#SBATCH --error=logs/job_%x_%A_%a.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=16Gb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-2 # One run per entry in SEEDS (src/config.py).

set -euo pipefail

CONFIG_FILE=${1:-configs/bp_common_input_pipeline.yaml}
EVAL_ARGS=${EVAL_ARGS:-}
RUN_ID=${SLURM_ARRAY_TASK_ID:-0}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1

LOGGER_ENABLED=$(python - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

import yaml

with Path(sys.argv[1]).open() as handle:
    config = yaml.safe_load(handle)

report_to_comet = config.get("evaluation", {}).get("report_to_comet")
logger_enabled = config.get("logger", {}).get("enabled", True)
effective_enabled = logger_enabled if report_to_comet is None else report_to_comet
print(str(effective_enabled).lower())
PY
)

if [[ "$LOGGER_ENABLED" == "true" && -z "${COMET_API_KEY:-}" ]]; then
    echo "ERROR: COMET_API_KEY must be exported when logger.enabled=true." >&2
    exit 1
fi

echo "Running hexel evaluation with config=${CONFIG_FILE}, run_id=${RUN_ID}"

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

python -m src.evaluate_hexels \
    --config="$CONFIG_FILE" \
    --run_id="$RUN_ID" \
    "${EVAL_ARG_ARRAY[@]}"
