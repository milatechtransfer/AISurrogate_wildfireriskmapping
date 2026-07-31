#!/bin/bash
##SBATCH --mail-type=all
 ##SBATCH --mail-user=<your_email>@<your_domain>
#SBATCH --job-name=baseline_xgb
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long-cpu
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=24Gb
#SBATCH --cpus-per-task=8

set -euo pipefail

CONFIG_FILE=${1:-configs/baselines/bp_spatial_only_xgb.yaml}
TRAIN_ARGS=${TRAIN_ARGS:-}

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

echo "Running baseline training with config: $CONFIG_FILE"
read -r -a TRAIN_ARG_ARRAY <<< "$TRAIN_ARGS"
python -m src.train_tabular_baseline --config="$CONFIG_FILE" "${TRAIN_ARG_ARRAY[@]}"
