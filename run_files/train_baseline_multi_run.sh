#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=name@mila.quebec
#SBATCH --job-name=baseline_xgb_multirun
#SBATCH --output=logs/job_%x_%A_%a.out
#SBATCH --error=logs/job_%x_%A_%a.err
#SBATCH --partition=long-cpu
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16Gb
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2                # Launches 3 parallel seeded runs (0,1,2); override e.g. `sbatch --array=0-4 ...` for 5 seeds.

set -euo pipefail

CONFIG_FILE=${1:-configs/bp_spatial_only_xgb.yaml}
TRAIN_ARGS=${TRAIN_ARGS:-}

RUN_ID=${SLURM_ARRAY_TASK_ID:-0}
echo "Starting run RUN_ID=${RUN_ID}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate

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

echo "Running baseline training with config: $CONFIG_FILE, run_id=$RUN_ID"
read -r -a TRAIN_ARG_ARRAY <<< "$TRAIN_ARGS"
python -m src.train_baseline \
    --config="$CONFIG_FILE" \
    --run_id="$RUN_ID" \
    "${TRAIN_ARG_ARRAY[@]}"