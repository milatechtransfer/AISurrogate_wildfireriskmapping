#!/bin/bash
#SBATCH --job-name=context_model
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=2-12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

: "${CONFIG:?CONFIG must point to a training YAML}"

if [[ -z "${COMET_API_KEY:-}" ]]; then
    echo "ERROR: COMET_API_KEY must be exported." >&2
    exit 1
fi

requeue_before_timeout() {
    echo "[SLURM] Requeueing ${SLURM_JOB_ID} before walltime; training will resume from last.pth."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap requeue_before_timeout USR1

train_args=(--config="$CONFIG")
if [[ "${SKIP_FINAL_HEXEL_ARTIFACTS:-0}" == "1" ]]; then
    train_args+=(--no_log_test_predicted_hexels --no_log_val_predicted_hexels)
fi

python -m src.train "${train_args[@]}" &
train_pid=$!
wait "$train_pid"
