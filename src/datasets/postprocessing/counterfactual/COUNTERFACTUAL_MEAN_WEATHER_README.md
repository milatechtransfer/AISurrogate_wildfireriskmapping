# Counterfactual Mean-Weather Intervention

Evaluate how BP, FI, and ROS predictions for hex16 change when its weather-zone
inputs are replaced by the mean processed-weather vector from hex17. Trained
checkpoints and all non-weather inputs remain fixed.

## Intervention

The configured `external_mean_zone_transplant` mode:

1. Reconstructs the raw weather-table row order with a hex ID attached to each row.
2. Computes the mean of every processed weather feature across all hex17 rows.
3. Builds the baseline `(hex_id, WeatherZone) -> feature vector` lookup table.
4. Replaces the five `(hex16, WeatherZone)` entries with the exact hex17 mean vector.

The materialized CSV contains one row per `(hex_id, WeatherZone)`. Entries for
non-recipient hexels retain their own baseline means, including when they share a
weather zone with hex16. Missing or unmatched zone pixels retain the baseline
per-hex mean fallback vector from the original weather table.

The configured donor has 136,692 weather rows. Its raw mean FWI is approximately
29.08, compared with approximately 19.27 across hex16's rows.

## Files

```text
configs/counterfactual_mean_weather.yaml
run_files/counterfactual_mean_weather_iROS.sh
run_files/counterfactual_mean_weather_plots.sh
src/evaluate_counterfactual.py
src/datasets/postprocessing/counterfactual/
  counterfactual_weather.py
  weather_counterfactual_transform.py
  plotting/counterfactual_response_maps.py
```

Evaluation writes:

```text
experiments/counterfactual_mean_weather_hex16/
  predictions/
    baseline/{bp,fi,ros}/
    bc_mean_weather_transplant/{bp,fi,ros}/
  figures/
    bc_mean_weather_transplant_{bp,fi,ros}/
  scenario_prediction_index.csv
  counterfactual_metrics.csv
  weather_edit_summary.csv
```

Each scenario prediction directory also contains the compact
`weather_intervention/weather_table_processed.csv` used for inference. It includes
both `hex_id` and `WeatherZone` lookup columns.

## Configuration

```yaml
raw_data_dir: "/path/to/raw/hexel/data"
save_dir: "experiments/counterfactual_mean_weather_hex16"
hex_ids: ["16"]
nonfuel_ids: [100, 101, 102, 105, 106, 110]

endpoints:
  bp:
    config_path: "configs/bp_common_input_pipeline.yaml"
  fi:
    config_path: "configs/fi_common_input_pipeline.yaml"
  ros:
    config_path: "configs/ros_common_input_pipeline.yaml"

scenarios:
  - name: "baseline"
    kind: "baseline"
    description: "Unmodified prepared weather inputs."

  - name: "bc_mean_weather_transplant"
    kind: "weather"
    description: "Assign every hex16 weather zone the exact mean processed-weather vector across all hex17 rows."
    params:
      mode: "external_mean_zone_transplant"
      donor_hex_ids: ["17"]
```

> Same as the fuel counterfactual config: if `bp`/`fi`/`ros` come from a single
> multi-output checkpoint instead of three separate models, point all three endpoints at
> that same `config_path` (see `configs/counterfactual_mean_weather_multi_output.yaml`
> for a ready-to-run example, or `configs/counterfactual_fuel_multi_output.yaml` for the
> equivalent fuel-scenario config); `evaluate_counterfactual.py` and the plotting scripts
> below handle this transparently.

`mode` is required explicitly. The current workflow supports only
`external_mean_zone_transplant`.

Each endpoint's `spatialized_weather` input source must set
`hex_id_col: "hex_id"` so inference resolves the materialized table by
`(hex_id, WeatherZone)` rather than pooling rows from hexels that share a zone.

## Running

Run all configured endpoints:

```bash
python -m src.evaluate_counterfactual \
  --config configs/counterfactual_mean_weather.yaml \
  --endpoint bp \
  --endpoint fi \
  --endpoint ros \
  --overwrite
```

Generate one response-map set:

```bash
python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_response_maps \
  --config configs/counterfactual_mean_weather.yaml \
  --scenario bc_mean_weather_transplant \
  --endpoint fi \
  --hex_id 16
```

On SLURM, submit `run_files/counterfactual_mean_weather_iROS.sh`, followed by
`run_files/counterfactual_mean_weather_plots.sh` with an `afterok` dependency.
