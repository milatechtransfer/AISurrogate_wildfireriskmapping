#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=name@mila.quebec
#SBATCH --job-name=unet_full_data
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=15:59:00
#SBATCH --mem=64Gb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue
#SBATCH --signal=B:TERM@300

set -euo pipefail

echo "Job has been requeued/restarted ${SLURM_RESTART_COUNT:-0} time(s)."

# Capture the first argument, default to 'configs/default_v1.yaml' if empty
CONFIG_FILE=${1:-configs/bp_common_input_pipeline.yaml}
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

RUN_CONFIG_FILE="$CONFIG_FILE"
ORIGINAL_DATA_ROOT_DIR=""
STAGE_DATA_TO_TMPDIR=${STAGE_DATA_TO_TMPDIR:-0}

if [[ "$STAGE_DATA_TO_TMPDIR" == "1" && -n "${SLURM_TMPDIR:-}" ]]; then
    echo "Using SLURM_TMPDIR for staged dataset: ${SLURM_TMPDIR}"

    ORIGINAL_DATA_ROOT_DIR=$(python - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
with config_path.open() as handle:
    config = yaml.safe_load(handle)

print(Path(config["data"]["root_dir"]).resolve())
PY
)

    STAGE_PARENT="${SLURM_TMPDIR}/nrcan_wildfireriskmapping_data"
    STAGED_DATA_ROOT_DIR="${STAGE_PARENT}/$(basename "$ORIGINAL_DATA_ROOT_DIR")"
    mkdir -p "$STAGE_PARENT"

    echo "Staging data.root_dir:"
    echo "  from: ${ORIGINAL_DATA_ROOT_DIR}"
    echo "  to:   ${STAGED_DATA_ROOT_DIR}"
    df -h "$SLURM_TMPDIR" || true

    if command -v rsync >/dev/null 2>&1; then
        rsync -a "${ORIGINAL_DATA_ROOT_DIR}/" "${STAGED_DATA_ROOT_DIR}/"
    else
        mkdir -p "$STAGED_DATA_ROOT_DIR"
        cp -a "${ORIGINAL_DATA_ROOT_DIR}/." "$STAGED_DATA_ROOT_DIR/"
    fi

    echo "Staged dataset size:"
    du -sh "$STAGED_DATA_ROOT_DIR" || true

    RUN_CONFIG_FILE="${SLURM_TMPDIR}/$(basename "${CONFIG_FILE%.yaml}")_slurm_tmpdir.yaml"
    python - "$CONFIG_FILE" "$RUN_CONFIG_FILE" "$STAGED_DATA_ROOT_DIR" <<'PY'
import sys
from pathlib import Path

import yaml

source_config = Path(sys.argv[1])
run_config = Path(sys.argv[2])
staged_root = Path(sys.argv[3])

with source_config.open() as handle:
    config = yaml.safe_load(handle)

config["data"]["root_dir"] = str(staged_root)

with run_config.open("w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)

print(f"Wrote staged config: {run_config}")
print(f"Using persistent raw_data_dir for normalization/evaluation: {config['data']['raw_data_dir']}")
PY
else
    echo "Using config data.root_dir directly."
fi

echo "Running training with config: $RUN_CONFIG_FILE"
read -r -a TRAIN_ARG_ARRAY <<< "$TRAIN_ARGS"
# Run as a proper SLURM job step (srun) rather than a plain child process so that
# --signal/--requeue reliably reach the training process and resource usage is
# accounted for correctly.
python -m src.train --config="$RUN_CONFIG_FILE" "${TRAIN_ARG_ARRAY[@]}"

if [[ -n "$ORIGINAL_DATA_ROOT_DIR" ]]; then
    echo "Restoring checkpoint config data.root_dir to persistent path: ${ORIGINAL_DATA_ROOT_DIR}"
    python - "$RUN_CONFIG_FILE" "$ORIGINAL_DATA_ROOT_DIR" <<'PY'
import sys
from pathlib import Path

import torch
import yaml

run_config = Path(sys.argv[1])
original_root = sys.argv[2]

with run_config.open() as handle:
    config = yaml.safe_load(handle)

save_dir = Path(config["save_dir"])
for checkpoint_name in ("best.pth", "last.pth"):
    checkpoint_path = save_dir / checkpoint_name
    if not checkpoint_path.exists():
        continue
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["config"]["data"]["root_dir"] = original_root
    torch.save(checkpoint, checkpoint_path)
    print(f"Updated {checkpoint_path}")
PY
fi

if [[ "$RUN_HEXEL_EVAL" == "1" ]]; then
    echo "Running evaluation with config: $RUN_CONFIG_FILE"
    read -r -a EVAL_ARG_ARRAY <<< "$EVAL_ARGS"
    python -m src.evaluate_hexels --config="$RUN_CONFIG_FILE" "${EVAL_ARG_ARRAY[@]}"
else
    echo "Skipping hexel evaluation because RUN_HEXEL_EVAL=${RUN_HEXEL_EVAL}"
fi
