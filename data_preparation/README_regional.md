# Data preparation pipeline

## ALWAYS UPDATE configs/CONFIG_FILE.yaml with the correct configurations under "data_prep"

Treat the regional data as one hexel, format it with the name: hexNN, and make sure files are structured as spatial/tabular/results.

Step 1: Process hexel data into multiple square patches, which will be our data samples:

You can run the following on an interactive node:
```bash
python -m data_preparation.process_hexels_into_grids --root_dir="/path/to/data/regional_data"  --save_dir="/path/to/data/regional_data/regional_data_scenario_A" --modelling_approach=1 --win_h=256 --win_w=256 --overlap_ratio=0.2 --ignition_weighting="distribution" --fuel_grid_representation="raw" --scenario_name="SCENARIO_NAME"
```

`--ignition_weighting` controls the ignition channels: `distribution` (default) produces zone-area-weighted 2-channel ignition (human + lightning), while `max` produces the original single-channel max-aggregation.

`--fuel_grid_representation` controls the fuel grid representation: `raw` (default) produces raw class values, while `group` groups similar classes together using `FUEL_GROUP_MAP` and saves the grid as 0-N values.

`--scenario_name="SCENARIO_NAME"` is used in case of regional/non-national data with multiple scenarios, this matters for fuel rasters as well as output rasters.

Step 2: Create training, validation and test splits.

```bash
python -m data_preparation.split_data --data_dir="/path/to/data/regional_data/regional_data_scenario_A" --test_only
```

Step 3: Create tabular files (weather + fire-size)

To produce the sequential weather table and the fire-size distribution table used by the model. Both tables are mapped to patches via the fire weather zone ID, so your grids must include that ID.

For fire size data, you should download "df_fire_fru.csv" from drive project folder / data (in case it is not already there in the root data folder on the cluster).

To build the tabular files, run the following:

First copy `fire_size_norm_params.json` and `weather_norm_params.json` from save_dir of the national data to new save_dir, then run:

```bash
python -m data_preparation.process_tabular_data \
	--root_dir="/path/to/data/regional_data" \
	--save_dir="/path/to/data/regional_data/regional_data_scenario_A" \
	--weather_output_file="weather_table_processed.csv" \
	--fire_size_input_file="ObservedFiresizedistribution_FortSimpson.csv" \
	--fire_size_output_file="ObservedFiresizedistribution_FortSimpson_processed.csv" \
	--modelling_approach=1 \
    --fire_size_norm_params_file="fire_size_norm_params.json" \
    --weather_norm_params_file="weather_norm_params.json"
```

Step 4 (necessary if fuel_grid_representation is '`raw`): Generate iROS values from the FBP package

To include new fuel classes other than the ones in `fuel_types_national.csv`, copy the file, and include the new classes. Save as a new file. For regional data, it is already saved under: `fuel_types_national_and_NWT.csv`
Do it locally and copy to the cluster (easier R support and we don't need access to all data to generate it)
`python -m data_preparation.tabular.fuel_features.generate_fuel_vectors_national --output-dir /path/to/data/regional_data/regional_data_scenario_A --fuel_types data_preparation/tabular/fuel_features/fuel_types_national_and_NWT.csv --output-filename=fbp_curves_national_and_NWT_fuel.csv` .

Step 5:

Copy `dataset_norm_stats.json` as `dataset_norm_stats_national.json` into the `save_dir` based on national data. If file is not there, see step 5 under `README.md`

To re-generate normalization stats for new regional data (output only - if needed for calibration): output local normalization stats can be are recomputed by scanning the raw rasters on every run.

```bash
python -m data_preparation.compute_dataset_normalization_stats \
	--raw_data_dir="/path/to/data/regional_data" \
	--root_dir="/path/to/data/regional_data/regional_data_scenario_A" \
	--save_dir="/path/to/data/regional_data/regional_data_scenario_A" \
	--train_split="test_indices.csv" \
	--types fire_intensity fire_ros fire_burn_probability
	--overwrite --scenario_name="SCENARIO_NAME"
```
