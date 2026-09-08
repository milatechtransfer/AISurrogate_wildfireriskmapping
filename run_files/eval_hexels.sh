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

set -euo pipefail

# Capture the first argument, default to the common pipeline config.
CONFIG_FILE=${1:-configs/bp_common_input_pipeline.yaml}
# Example: EVAL_ARGS="--no_log_test_predicted_hexels"
EVAL_ARGS=${EVAL_ARGS:-}

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

echo "Using data.root_dir directly from config: $CONFIG_FILE"

echo "Running hexel evaluation with config: $CONFIG_FILE"
read -r -a EVAL_ARG_ARRAY <<< "$EVAL_ARGS"
python -m src.evaluate_hexels \
    --config="$CONFIG_FILE" \
    "${EVAL_ARG_ARRAY[@]}"
