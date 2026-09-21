# Counterfactual Fire-Size Intervention

Evaluate how BP, FI, and ROS predictions for hex16 change when its q10/q50/q90
fire-size inputs are shifted to reflect a change in how long fires spread. Trained
checkpoints and all other inputs remain fixed.

## Intervention

The configured `spread_day_quantile_scaling` mode holds the ignition-derived
seasonal likelihoods fixed and edits fire size only:

1. Reads `hex<ID>_IgnitionDistribution.csv` and aggregates cause-specific relative
   likelihoods into `P(season | fire zone)`.
2. Reads the `Spread Event Distribution - fruNN - sN` rows from
   `hex<ID>_ScenarioDistributions - * - FINAL.csv` as per-`(fire zone, season)`
   discrete PMFs over spread-event days.
3. Forms each fire zone's exact seasonal mixture PMF and takes its inverse-CDF
   q50 and q90 spread-day values.
4. Adds `spread_day_delta_q50_days` / `spread_day_delta_q90_days` to those
   spread-day quantiles and converts the ratios into fire-size multipliers via
   `((d + delta) / d) ** size_scaling_exponent`.
5. Applies those multipliers to the corresponding baseline fire-size quantiles in
   hectares, then re-normalizes back into the model's `NORM_LOG_SIZE_HA` feature
   scale using `fire_size_norm_params.json`.

The deltas are signed and independent. Positive values lengthen spread events
(multiplier > 1, larger fires); negative values shorten them (multiplier < 1,
smaller fires), which is how you'd express, say, improved suppression response
rather than a worsening climate. Setting one delta to `0` leaves that quantile
unchanged, so you can reshape the distribution rather than only shifting it.

q10 is never scaled: the mode targets the middle and upper tail of the fire-size
distribution. Two guards apply to every `(hex, fire zone)`:

- `d + delta` must stay positive, so a negative delta cannot drive a spread-day
  quantile to zero or below.
- the resulting quantiles must still satisfy `q10 <= q50 <= q90`.

Both raise rather than silently producing a degenerate distribution, so aggressive
deltas fail loudly.

`size_scaling_exponent` is an explicit growth sensitivity assumption (`beta = 2`
means area scales with the square of spread duration, i.e. roughly radial growth),
not a fitted coefficient.

Rather than editing the source table in place, the scenario writes exact,
hex-scoped, precomputed quantile columns and repoints the `spatialized_fire_size`
input source at them, so inference consumes the intervened quantiles directly
instead of recomputing them.

## Files

```text
configs/counterfactual/counterfactual_fire_size_spread_days_multi_output.yaml
run_files/counterfactual/counterfactual_fire_size_spread_days_iROS.sh
run_files/counterfactual/counterfactual_fire_size_spread_days_plots.sh
src/evaluate_counterfactual.py
src/datasets/postprocessing/counterfactual/
  fire_size_counterfactual_transform.py
  plotting/counterfactual_response_maps.py
```

Evaluation writes:

```text
experiments/counterfactual_fire_size_spread_days_multi_output_hex16_q3_256/
  predictions/
    baseline/{bp,fi,ros}/
    spread_days_q50_plus_0p7_q90_plus_5_beta2/{bp,fi,ros}/
  figures/
  scenario_prediction_index.csv
  counterfactual_metrics.csv
  fire_size_edit_summary.csv
```

Each scenario prediction directory also contains
`fire_size_intervention/fire_size_quantile_intervention.csv` (the
`(hex_id, GRIDCODE) -> q10/q50/q90` lookup used for inference) and
`fire_size_quantile_global_fill.csv` (the per-hex fallback for unmatched zones).

`fire_size_edit_summary.csv` records, per `(hex_id, fire zone)`: the seasonal
weights, mixture spread-day q50/q90, the derived multipliers, and baseline vs
future fire sizes in both hectares and normalized feature units.

## Configuration

```yaml
raw_data_dir: "/path/to/raw/hexel/data"
save_dir: "experiments/counterfactual_fire_size_spread_days_multi_output_hex16"
hex_ids: ["16"]
nonfuel_ids: [100, 101, 102, 105, 106, 110]

endpoints:
  bp:
    config_path: "configs/multi_output_spatial_weather_firesize_q3.yaml"
    checkpoint_dir: "/network/projects/amlrt/nrcan_wildfires/checkpoints/burnp3plus/final_experiments/unet_256_firesize_q3"
  # fi/ros alias the same multi-output checkpoint

scenarios:
  - name: "baseline"
    kind: "baseline"
    description: "Unmodified q10/q50/q90 fire-size inputs."

  - name: "spread_days_q50_plus_0p7_q90_plus_5_beta2"
    kind: "fire_size"
    params:
      mode: "spread_day_quantile_scaling"
      spread_day_delta_q50_days: 0.7
      spread_day_delta_q90_days: 5.0
      size_scaling_exponent: 2.0
      pmf_total_tolerance_percent: 1.0
```

To shorten spread events instead, use negative deltas, e.g.
`spread_day_delta_q90_days: -2.0`.

The endpoint config's `spatialized_fire_size` source must declare exactly one
feature (`NORM_LOG_SIZE_HA`) with `quantiles: [0.1, 0.5, 0.9]`; other quantile sets
are rejected.

`pmf_total_tolerance_percent` (default `1.0`) bounds how far each source spread
distribution may deviate from summing to 100%, guarding against truncated or
malformed input tables.

## Running

```bash
python -m src.evaluate_counterfactual \
  --config configs/counterfactual/counterfactual_fire_size_spread_days_multi_output.yaml \
  --endpoint bp --endpoint fi --endpoint ros --overwrite
```

Generate one response-map set:

```bash
python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_response_maps \
  --config configs/counterfactual/counterfactual_fire_size_spread_days_multi_output.yaml \
  --scenario spread_days_q50_plus_0p7_q90_plus_5_beta2 \
  --endpoint fi \
  --hex_id 16
```

The default SLURM workflow evaluates all three configured seeds and aggregates their
mean/std responses:

```bash
bash run_files/counterfactual/submit_all_counterfactuals.sh fire_size
```

Use the same entry point with `--single-seed` for a seed-42-only fallback.
