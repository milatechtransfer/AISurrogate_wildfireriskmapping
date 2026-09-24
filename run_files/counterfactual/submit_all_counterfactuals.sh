#!/bin/bash
# Submit all three seeded q3 evaluations, per-seed plots, and final mean/std
# aggregation for the four retained counterfactual scenarios.
#
# Usage:
#   bash run_files/counterfactual/submit_all_counterfactuals.sh [--dry-run] [--single-seed] [name ...]
#
# Names: fuel, fuel_polygons, mean_weather, fire_size
# By default, run_ids 0-2 select seeds 42, 1337, and 2024. `--single-seed`
# submits run_id 0 (seed 42) and skips across-seed aggregation.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
mkdir -p logs

dry_run=0
single_seed=0
requested=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run) dry_run=1 ;;
        --single-seed) single_seed=1 ;;
        -*) echo "Unknown flag: ${arg}" >&2; exit 2 ;;
        *) requested+=("${arg}") ;;
    esac
done

array_spec="0-2"
if [[ ${single_seed} -eq 1 ]]; then
    array_spec="0"
fi

# name : config : scenario : scenario kind
EXPERIMENTS=(
    "fuel:configs/counterfactual/counterfactual_fuel_multi_output.yaml:c2_to_mixedwood_fixed:fuel"
    "fuel_polygons:configs/counterfactual/counterfactual_fuel_polygons_multi_output.yaml:pooled_burn_scars_to_aspen:fuel"
    "mean_weather:configs/counterfactual/counterfactual_mean_weather_multi_output.yaml:bc_mean_weather_transplant:weather"
    "fire_size:configs/counterfactual/counterfactual_fire_size_spread_days_multi_output.yaml:spread_days_q50_plus_0p7_q90_plus_5_beta2:fire_size"
)

selected=()
for entry in "${EXPERIMENTS[@]}"; do
    name="${entry%%:*}"
    if [[ ${#requested[@]} -eq 0 ]]; then
        selected+=("${entry}")
        continue
    fi
    for want in "${requested[@]}"; do
        if [[ "${want}" == "${name}" ]]; then
            selected+=("${entry}")
        fi
    done
done

if [[ ${#selected[@]} -eq 0 ]]; then
    echo "No matching experiments for: ${requested[*]}" >&2
    exit 2
fi

run_dir="run_files/counterfactual"
eval_script="${run_dir}/counterfactual_multiseed_eval.sh"
plot_script="${run_dir}/counterfactual_multiseed_plots.sh"
aggregate_script="${run_dir}/counterfactual_multiseed_aggregate.sh"
for script in "${eval_script}" "${plot_script}" "${aggregate_script}"; do
    if [[ ! -f "${script}" ]]; then
        echo "Missing script: ${script}" >&2
        exit 1
    fi
done
for entry in "${selected[@]}"; do
    IFS=':' read -r _ config _ _ <<<"${entry}"
    if [[ ! -f "${config}" ]]; then
        echo "Missing config: ${config}" >&2
        exit 1
    fi
done

printf '%-16s %-12s %-12s %-12s\n' "EXPERIMENT" "EVAL_JOB" "PLOT_JOB" "AGG_JOB"
for entry in "${selected[@]}"; do
    IFS=':' read -r name config scenario scenario_kind <<<"${entry}"
    exports="ALL,CONFIG=${config},SCENARIO=${scenario},SCENARIO_KIND=${scenario_kind}"

    if [[ ${dry_run} -eq 1 ]]; then
        echo "sbatch --array=${array_spec} --job-name=cf_${name}_eval --export=${exports} ${eval_script}"
        echo "sbatch --array=${array_spec} --dependency=afterok:<eval> --job-name=cf_${name}_plots --export=${exports} ${plot_script}"
        if [[ ${single_seed} -eq 0 ]]; then
            echo "sbatch --dependency=afterok:<eval> --job-name=cf_${name}_agg --export=${exports} ${aggregate_script}"
            aggregate_job="DRYRUN"
        else
            aggregate_job="SKIPPED"
        fi
        printf '%-16s %-12s %-12s %-12s\n' "${name}" "DRYRUN" "DRYRUN" "${aggregate_job}"
        continue
    fi

    eval_job=$(sbatch --parsable --array="${array_spec}" --job-name="cf_${name}_eval" --export="${exports}" "${eval_script}")
    eval_job="${eval_job%%;*}"
    plot_job=$(
        sbatch --parsable \
            --array="${array_spec}" \
            --dependency="afterok:${eval_job}" \
            --kill-on-invalid-dep=yes \
            --job-name="cf_${name}_plots" \
            --export="${exports}" \
            "${plot_script}"
    )
    plot_job="${plot_job%%;*}"
    if [[ ${single_seed} -eq 0 ]]; then
        aggregate_job=$(
            sbatch --parsable \
                --dependency="afterok:${eval_job}" \
                --kill-on-invalid-dep=yes \
                --job-name="cf_${name}_agg" \
                --export="${exports}" \
                "${aggregate_script}"
        )
        aggregate_job="${aggregate_job%%;*}"
    else
        aggregate_job="SKIPPED"
    fi
    printf '%-16s %-12s %-12s %-12s\n' "${name}" "${eval_job}" "${plot_job}" "${aggregate_job}"
done
