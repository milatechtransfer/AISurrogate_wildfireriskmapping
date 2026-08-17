# Counterfactual Weather Interventions

Evaluate how BP, FI, and ROS predictions for hex16 change when its weather-zone
inputs are edited via one of three interventions: transplanting hex17's mean
processed-weather vector, transplanting the mean processed-weather vector of
hex16's own windiest days, or sweeping hex16's wind through 9 fixed compass
directions. Trained checkpoints and all non-weather inputs remain fixed.

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

`mode` is required explicitly. The current workflow supports
`external_mean_zone_transplant` and `windy_mean_zone_transplant` (below).

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

## Windy Self-Transplant Intervention

`configs/counterfactual/counterfactual_windy_weather.yaml` (and its
`_multi_output` counterpart) evaluate a second weather edit: give hex16 the mean
processed-weather vector of hex16's own windiest days, instead of an external
donor hexel. This isolates the effect of a hexel's high-wind conditions becoming
its typical conditions, without mixing in a different hexel's climate.

The configured `windy_mean_zone_transplant` mode:

1. Reconstructs the raw weather-table row order with a hex ID attached to each row.
2. Restricts the donor rows to `donor_hex_ids` rows whose raw `WindSpeed` is `>=`
   the configured `wind_speed_threshold`.
3. Computes the mean of every processed weather feature across those windy donor
   rows only.
4. Builds the baseline `(hex_id, WeatherZone) -> feature vector` lookup table.
5. Replaces the recipient hexels' `(hex_id, WeatherZone)` entries with the exact
   windy-donor mean vector.

`donor_hex_ids` may equal `recipient_hex_ids` (a hexel donates its own windy rows
to itself, as in the example config below) or differ (an external hexel's windy
rows are transplanted, as with `external_mean_zone_transplant`).

The example config uses `donor_hex_ids: ["16"]`, `wind_speed_threshold: 20`
(km/h) - approximately the 90th percentile of hex16's raw `WindSpeed`
distribution (median 11, 90th pct 20, 95th pct 23.4, max 64.8), so the donor mean
is computed over hex16's windiest ~10% of rows.

### Configuration

```yaml
raw_data_dir: "/path/to/raw/hexel/data"
save_dir: "experiments/counterfactual_windy_weather_hex16"
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

  - name: "bc_windy_self_transplant"
    kind: "weather"
    description: "Assign every hex16 weather zone the exact mean processed-weather vector across hex16's own rows with raw WindSpeed >= 20 km/h."
    params:
      mode: "windy_mean_zone_transplant"
      donor_hex_ids: ["16"]
      wind_speed_threshold: 20
```

`wind_speed_threshold` is required for this mode (a numeric raw `WindSpeed`
lower bound, applied as `>=`) and raises a clear `ValueError` if omitted, or if no
donor rows clear the threshold.

### Running

Run all configured endpoints:

```bash
python -m src.evaluate_counterfactual \
  --config configs/counterfactual/counterfactual_windy_weather_multi_output.yaml \
  --endpoint bp \
  --endpoint fi \
  --endpoint ros \
  --overwrite
```

Generate one response-map set:

```bash
python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_response_maps \
  --config configs/counterfactual/counterfactual_windy_weather_multi_output.yaml \
  --scenario bc_windy_self_transplant \
  --endpoint fi \
  --hex_id 16
```

## Wind-Direction Sweep Intervention

`configs/counterfactual/counterfactual_wind_direction_multi_output.yaml` sweeps a
single hexel's wind through 9 compass directions (0/45/.../315, plus 360 = 0 again
to close the loop for circular/rose plots), holding every donor row's real
`WindSpeed` magnitude fixed and only forcing its direction. Non-wind features
(temperature, humidity, FFMC, FWI, ...) are still averaged from the real donor
rows, unchanged.

The configured `wind_direction_zone_transplant` mode:

1. Restricts donor rows to `donor_hex_ids` rows whose raw `WindSpeed` is `>=`
   `wind_speed_threshold` (set the threshold to `0` to use every donor row / the
   donor's ordinary average wind speed, instead of only its windiest days).
2. For each surviving donor row, keeps its recorded `WindSpeed` but overrides its
   `WindDirection` to the scenario's `direction_degrees`, then recomputes
   `wind_x`/`wind_y` and re-normalizes them with the same z-score parameters
   fit at training time (`weather_norm_params.json`, expected alongside the
   processed weather table).
3. Averages every other weather feature unchanged across the same donor rows.
4. Replaces the recipient hexels' `(hex_id, WeatherZone)` entries with that vector.

**Prefer a self-donor** (`donor_hex_ids == recipient_hex_ids`, as in the example
below) to isolate the pure direction effect: the recipient keeps its own real
non-wind climate, and only wind direction is swept. An external donor also
imports that donor's non-wind climate, conflating a wind-direction sweep with a
mean-weather transplant.

Because all 9 direction scenarios share one `save_dir`,
`evaluate_counterfactual.py` and the plotting script write each direction's
predictions/figures to its own `predictions/wind_dir_XXX/` and
`figures/wind_dir_XXX_<target>/` subfolder automatically - no extra scaffolding
needed for "one experiment folder, one subfolder per direction".

### Configuration

```yaml
raw_data_dir: "/path/to/raw/hexel/data"
save_dir: "experiments/counterfactual_wind_direction_hex16"
hex_ids: ["16"]
nonfuel_ids: [100, 101, 102, 105, 106, 110]

endpoints:
  bp:
    config_path: "configs/multi_output_spatial_weather.yaml"
  fi:
    config_path: "configs/multi_output_spatial_weather.yaml"
  ros:
    config_path: "configs/multi_output_spatial_weather.yaml"

scenarios:
  - name: "baseline"
    kind: "baseline"
    description: "Unmodified prepared weather inputs."

  - name: "wind_dir_000"
    kind: "weather"
    description: "Hex16's average wind speed (all rows, WindSpeed >= 0), wind forced to blow from 0 deg (N)."
    params: {mode: "wind_direction_zone_transplant", donor_hex_ids: ["16"], wind_speed_threshold: 0, direction_degrees: 0}
  # ... one scenario per direction (045, 090, 135, 180, 225, 270, 315, 360)
```

`direction_degrees` and `wind_speed_threshold` are both required for this mode
and raise a clear `ValueError` if omitted (`wind_speed_threshold: 0` includes
every donor row, i.e. the donor's ordinary average wind speed); the mode also
requires `weather_norm_params.json` to exist next to the processed weather
table.

### Running

Run all configured endpoints:

```bash
python -m src.evaluate_counterfactual \
  --config configs/counterfactual/counterfactual_wind_direction_multi_output.yaml \
  --endpoint bp \
  --endpoint fi \
  --endpoint ros \
  --overwrite
```

Generate the response maps for every direction:

```bash
for scenario in wind_dir_000 wind_dir_045 wind_dir_090 wind_dir_135 wind_dir_180 wind_dir_225 wind_dir_270 wind_dir_315 wind_dir_360; do
  python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_response_maps \
    --config configs/counterfactual/counterfactual_wind_direction_multi_output.yaml \
    --scenario "${scenario}" \
    --endpoint fi \
    --hex_id 16
done
```

Generate a compass-rose figure per endpoint - the 8 unique-bearing \u0394 maps (last
panel of the response maps above) arranged on a circle, 0\u00b0 at the top (north)
and bearings increasing clockwise (45\u00b0 = NE, 90\u00b0 = E, ...), on one shared
colour scale for direct visual comparison:

```bash
python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_wind_direction_compass \
  --config configs/counterfactual/counterfactual_wind_direction_multi_output.yaml \
  --endpoint fi \
  --hex_id 16
```

Written to `figures/compass/wind_direction_compass_<endpoint>.png`.

On SLURM, submit `run_files/counterfactual/counterfactual_wind_direction_iROS.sh`,
followed by `run_files/counterfactual/counterfactual_wind_direction_plots.sh` (which
also generates the compass figures) with an `afterok` dependency.
