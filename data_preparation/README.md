# Data preparation pipeline

## ALWAYS UPDATE configs/CONFIG_FILE.yaml with the correct configurations under "data_prep"

Step 1: Process hexel data into multiple square patches, which will be our data samples:

Note: If you want to run this in the cluster using SLURM array jobs, you can modify the `run_files/generate_grid_data.sh` by changing the save directory path and run the following in the terminal (from the main directory)

```bash
sbatch run_files/generate_grid_data.sh
```

Instead, you can run the following on an interactive node:
```bash
python -m data_preparation.process_hexels_into_grids --root_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA"  --save_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v4" --modelling_approach=1 --win_h=256 --win_w=256 --overlap_ratio=0.2 --ignition_weighting="distribution" --fuel_grid_representation="raw"
```

`--ignition_weighting` controls the ignition channels: `distribution` (default) produces zone-area-weighted 2-channel ignition (human + lightning), while `max` produces the original single-channel max-aggregation.

`--fuel_grid_representation` controls the fuel grid representation: `raw` (default) produces raw class values, while `group` groups similar classes together using `FUEL_GROUP_MAP` and saves the grid as 0-N values.

Step 2: Create training, validation and test splits.

- Run the `get_stratified_data_split` function in `data_preparation/utils.py` to run stratified sampling over the available hex_ids. This will give a train, val, test split with 37,5,5 hexels in each respectively
- Now, verify this split looking at the geographical map and find if the split is well distributed
- Once the split is finalized, use the decided split to obtain the train/val/test csvs
- Finally, run the following with the decided splits

```bash
python -m data_preparation.split_data --data_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v4" --val_hex_id 02 23 33 18 46 --test_hex_id 01 12 39 16 49
```

Step 3: Create tabular files (weather + fire-size)

To produce the sequential weather table and the fire-size distribution table used by the model. Both tables are mapped to patches via the fire weather zone ID, so your grids must include that ID.

For fire size data, you should download "df_fire_fru.csv" from drive project folder / data (in case it is not already there in the root data folder on the cluster).

To build the tabular files, run the following:


```bash
python -m data_preparation.process_tabular_data \
	--root_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA" \
	--save_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v4" \
	--weather_output_file="weather_table_processed.csv" \
	--fire_size_input_file="df_fire_fru_25ha_1970_2023.csv" \
	--fire_size_output_file="df_fire_fru_processed.csv" \
	--train_split_file="train_indices.csv" \
	--modelling_approach=1
```
For national data, this saves : `weather_norm_params.json` and `fire_size_norm_params.json` inside `save_dir`

[Important] For new evaluation data (NWT data), pass `--fire_size_norm_params_file` and `--weather_norm_params_file` to read train-only normalization parameters previously computed from national study

Notes: If you used modelling approach 2, set `--save_dir` to `data_samples_approach_2`. The `process_tabular_data` script will look for the fire-size file in `--root_dir` first, then in `--save_dir`; ensure `df_fire_fru.csv` is present in one of those places.

Step 4 (necessary if fuel_grid_representation is '`raw`): Generate iROS values from the FBP package

Or do it locally and copy to the cluster (easier R support and we don't need access to all data to generate it)
`python -m data_preparation.tabular.fuel_features.generate_fuel_vectors_national --output-dir /network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v4 --fuel_types data_preparation/tabular/fuel_features/Fuel_Types.csv`, you can modify `Fuel_Types.csv` to include more fuel types.

This saves csv file in the root_dir called `fbp_curves_national_fuel.csv`.

Step 5 (for training data only, to be used by eval-only data): Precompute input and target normalization stats

The `log_standard` target normalization needs train-only log1p mean/std constants. The `min/max normalization` for burn probability and elevation needs train-only data, as well as fuel curves features. These are otherwise recomputed by scanning the raw rasters on every run; computing them once offline writes a `dataset_norm_stats.json` into the `save_dir` so training/eval/inference just read the cached values.

```bash
python -m data_preparation.compute_dataset_normalization_stats \
	--raw_data_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA" \
	--root_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v4" \
	--save_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v4" \
	--train_split="train_indices.csv" \
	--types elevation fuel_curve_iROS fuel_curve_HFI fire_intensity fire_ros fire_burn_probability
```
