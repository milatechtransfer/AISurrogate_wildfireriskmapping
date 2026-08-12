#!/bin/bash
set -euo pipefail

CONFIG_FILE=${1:-configs/runtime_benchmarks/multi_output_hex16.yaml}
N_REPS=${N_REPS:-5}
EVAL_ARGS=${EVAL_ARGS:-"--fast_eval"}

cd "$(dirname "$0")/../.."
mkdir -p logs
source .venv/bin/activate

echo "Using config: ${CONFIG_FILE}"
echo "Running ${N_REPS} repetitions for timing/memory variance (MPS)"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for REP in $(seq 1 "$N_REPS"); do
    LOG_FILE="logs/eval_mac_${TIMESTAMP}_rep${REP}.out"
    echo "=======Starting rep ${REP}/${N_REPS}========"

    read -r -a EVAL_ARG_ARRAY <<< "$EVAL_ARGS"
    python -m src.evaluate_hexels \
        --config="$CONFIG_FILE" \
        "${EVAL_ARG_ARRAY[@]}" \
        > "$LOG_FILE" 2>&1

    echo "Rep ${REP} finished -> ${LOG_FILE}"
done

echo "=======Benchmark Complete========"
grep -h "Device\] Using\|Total Evaluation Time\|Dataloader Setup Time\|Trainer Init Time\|Checkpoint Load Time\|Prediction Time\|Eval Time Excl. Dataloader Setup\|Peak Host RSS\|Peak GPU Reserved\|Peak GPU Allocated\|Peak MPS Driver Allocated\|Peak MPS Current Allocated\|MPS Driver Allocated\|MPS Current Allocated\|PostprocessTiming" \
    logs/eval_mac_${TIMESTAMP}_rep*.out \
    | tee logs/eval_mac_${TIMESTAMP}_summary.out
