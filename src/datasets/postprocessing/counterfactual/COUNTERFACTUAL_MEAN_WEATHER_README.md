# Counterfactual Weather Interventions

Evaluate how BP, FI, and ROS predictions for hex16 change when its weather-zone
inputs are edited: either replaced by the mean processed-weather vector from
hex17, or edited per-`WeatherZone` to isolate wind-speed and wind-direction
effects. Trained checkpoints and all non-weather inputs remain fixed.

## Mean-Weather Intervention

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

`mode` is required explicitly. Supported modes are `external_mean_zone_transplant`
(above) plus the two zone-dependent modes described below.

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

## Zone-Dependent Wind Interventions

`external_mean_zone_transplant` pools donor rows across the *whole* donor hex and
broadcasts one identical mean vector to *every* recipient `WeatherZone`, so after
the edit every zone within a hexel has exactly the same weather. That collapses
natural zone-to-zone heterogeneity (e.g. hex16's zone 4 averages ~16.4 km/h wind
vs zone 12's ~10.3 km/h).

The two zone-dependent modes instead group donor rows by
`(donor_hex_id, WeatherZone)`: each recipient zone gets a donor mean computed only
from donor rows sharing *that same* `WeatherZone`, so zones keep their own distinct
weather after the edit. `weather_edit_summary.csv` gains a `weather_zone` column and
carries one row per recipient `(hex_id, WeatherZone)` pair for these modes.

**Only use a self-donor** (`donor_hex_ids == recipient_hex_ids`) for these modes: an
external donor would additionally require its `WeatherZone` IDs to line up with the
recipient's, but zone IDs are assigned independently per hexel and are not guaranteed
to be spatially or climatically comparable across hexels.

### `windy_mean_zone_dependent_transplant`

Each recipient zone receives the mean processed-weather vector of its own windiest
rows. `wind_speed_percentile` is resolved separately within each zone's own raw
`WindSpeed` distribution - a single hex-wide cutoff would not isolate comparably
"windy" rows per zone. `90` keeps each zone's windiest 10%; lower it (e.g. `75`) to
widen the donor pool, raise it (e.g. `95`) to narrow it, and `0` keeps every donor
row (that zone's ordinary average).

```yaml
  - name: "windy_self_transplant_zone_dependent"
    kind: "weather"
    params:
      mode: "windy_mean_zone_dependent_transplant"
      donor_hex_ids: ["16"]
      wind_speed_percentile: 90
```

The optional `season: <int>` param (the processed `Season` integer, e.g. hex16's
s1/s2) is applied per-zone *before* that zone's percentile cutoff, so "windiest days"
is resolved within `(zone, season)` jointly:

```yaml
  - name: "windy_self_transplant_zone_dependent_s1"
    kind: "weather"
    params:
      mode: "windy_mean_zone_dependent_transplant"
      donor_hex_ids: ["16"]
      wind_speed_percentile: 90
      season: 1
```

### `wind_direction_zone_dependent_transplant`

Each recipient zone keeps its own same-zone wind magnitude but has `WindDirection`
overridden to `direction_degrees` before `wind_x`/`wind_y` are recomputed and
re-normalized with the z-score parameters fit at training time (read from
`weather_norm_params.json` next to the processed weather table). All other weather
columns are averaged unchanged. Use `wind_speed_percentile: 0` to sweep direction at
each zone's ordinary average wind speed rather than only its windiest days.

```yaml
  - name: "wind_dir_000"
    kind: "weather"
    params: {mode: "wind_direction_zone_dependent_transplant", donor_hex_ids: ["16"], wind_speed_percentile: 0, direction_degrees: 0}
  # ... one scenario per direction (045, 090, 135, 180, 225, 270, 315, 360)
```

`360` repeats `0` (N) so compass-rose plots close the loop.

### Running

```bash
python -m src.evaluate_counterfactual \
  --config configs/counterfactual/counterfactual_windy_weather_zone_dependent_multi_output.yaml \
  --endpoint bp --endpoint fi --endpoint ros --overwrite

python -m src.evaluate_counterfactual \
  --config configs/counterfactual/counterfactual_wind_direction_zone_dependent_multi_output.yaml \
  --endpoint bp --endpoint fi --endpoint ros --overwrite
```

Response maps reuse `counterfactual_response_maps` (pass `--zone_overlay` to draw
firezone boundaries). The direction sweep additionally renders compass roses - eight
direction-diff maps arranged around a circle, one figure per endpoint:

```bash
python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_wind_direction_compass \
  --config configs/counterfactual/counterfactual_wind_direction_zone_dependent_multi_output.yaml \
  --endpoint fi --hex_id 16 --zone_overlay
```

On SLURM, submit each `run_files/counterfactual/*_zone_dependent_iROS.sh` followed by
its `*_zone_dependent_plots.sh` with an `afterok` dependency.

## Shared Checkpoints

Endpoints resolve their checkpoint from the endpoint config's own `save_dir` by
default. Set `checkpoint_dir` on an endpoint to read a shared released checkpoint
instead, which is what the shipped zone-dependent configs do:

```yaml
endpoints:
  bp:
    config_path: "configs/multi_output_spatial_weather_firesize_q3.yaml"
    checkpoint_dir: "/network/projects/amlrt/nrcan_wildfires/checkpoints/burnp3plus/final_experiments/unet_256_firesize_q3"
```
