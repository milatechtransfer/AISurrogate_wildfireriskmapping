# Inference Module

End-to-end inference pipeline for wildfire risk prediction on hexels.

## Predict from a model bundle (recommended)

A model bundle is a self-contained folder with the weights, the training normalization statistics,
the fuel curves and a manifest (see `bundle.py`). Prediction from a bundle needs only the project
**inputs**; BurnP3+ outputs and the training data are not required.

```bash
# One-off (maintainers): turn a training checkpoint into a bundle
python -m inference.export_bundle --checkpoint path/to/best.pth --out_dir nrcan-surrogate-bp-fi-ros-v1.0 \
    --name nrcan-surrogate-bp-fi-ros --version 1.0.0 \
    --hazard_denominator_json path/to/hazard_scale_denominator.json \
    --fire_size_training_table path/to/df_fire_fru_25ha_1970_2023.csv

# Predict every hexel of a project (CPU or GPU is picked automatically)
python -m inference.predict --bundle nrcan-surrogate-bp-fi-ros-v1.0 --project path/to/project \
    --fire_size_table path/to/fire_sizes.csv --output path/to/predictions
```

Useful options: `--hex_ids 12 14`, `--device cpu|cuda|mps`, `--batch_size` (lower it if memory runs out),
`--mask_scope buffer`, `--no_hazard`, `--overwrite`, `--keep_work_dir`. Run with `--help` for the full list.

Project inputs per hexel (`hexNN/`): `spatial/hexNN_dem.tif` (100 m), `spatial/hexNN_fbp.tif`,
`spatial/hexNN_firezones.tif`, `spatial/ignition_grids/hexNN_ignGrid_{H|N}_<season>.tif`,
`spatial/mask_grids/hexNN_actual.shp`, and `tabular/hexNN_{DailyWeather,FireZones,IgnitionDistribution,GreenUp}.csv`.
The fire-size table (columns `GRIDCODE`, `SIZE_HA`) is supplied separately; the bundle manifest records
which table the model was trained with.

Outputs in `--output`: `hexNN/hexNN_{bp,fi,ros}.tif` (probability, kW/m, m/min; nodata -9999),
`hexNN/hexNN_hazard_{raw,scaled,class}.tif` (class 0 = nodata), `run_manifest.json` (bundle, inputs,
options, software versions, timings) and `predict.log`.

## Legacy: predict from a training checkpoint

The pipeline below reads normalization statistics and fuel curves from the training data folders
referenced by the checkpoint and needs BurnP3+ outputs for every hexel. Prefer `inference.predict`.

## Quick Start

### Command Line

Please run this from the root of the repository to ensure correct paths to data and checkpoints. The CLI accepts arguments that override values in `config.yaml` for flexibility. Below are example commands for different use cases:

```bash
# Option 1: Run inference using values from config.yaml
python -m inference.run_ai_surrogate_model_hexel_inference

# Option 2: Override hexel ID and data preparation
python -m inference.run_ai_surrogate_model_hexel_inference --hex_id="05" --prepare_data=False

# Option 3: Run inference on all hexels
python -m inference.run_ai_surrogate_model_hexel_inference --hex_id="all" --prepare_data=True

# Option 4: Run inference using all CLI commands
python -m inference.run_ai_surrogate_model_hexel_inference --data_dir=/path/to/hexel/data --checkpoint_path=/path/to/model/best.pth --save_dir=/path/to/output/ --hex_id="all" --batch_size=64 --num_workers=8 --prepare_data=True
```
**CLI Arguments**

Note: All arguments follow values from `config.yaml` but if CLI arguments are provided, they will override the config values.

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config file | `inference/config.yaml` |
| `--data_dir` | Directory containing hexel data | From config |
| `--checkpoint_path` | Path to model checkpoint | From config |
| `--save_dir` | Directory to save predictions and visualizations | From config |
| `--hex_id` | (str) Hexel ID (str) to process | From config |
| `--prepare_data` | Run data preparation step | `True` |
| `--batch_size` | Batch size for inference | From config |
| `--num_workers` | DataLoader workers | From config |



## Output

We assume `save_dir` is set to `outputs/` for the following paths. For each hexel, all the outputs are saved under corresponding subdirectories in `outputs/` with the hexel ID in the filename. For example, for hexel "02", the predicted patches will be saved as `outputs/predicted_patches/hexel_02.npy`.

````
inference/
└── outputs/
    ├── predicted_hexels/                    # Reconstructed hexel grid of predicted target values and geospatial profile from ground truth as GeoTIFF files
    ├── predicted_hexels_plot/               # Visualizations comparing ground truth vs predictions for each hexel
    └── predicted_patches/                   # Normalized patch predictions (N, C, H, W) as .npy files
````

## Configuration

Edit `config.yaml`:

```yaml
# Paths
data_dir: "/path/to/hexel/data"
checkpoint_path: "/path/to/model/best.pth"
save_dir: "/path/to/output/"  # Predictions saved as predictions_hexel_{hex_id}.npy

# Hexel configuration
hex_id: "02"

# Data preparation settings
prepare_data: True      # Set true to run data prep

# Inference settings
batch_size: 32
num_workers: 4
```

## Data Requirements

The `data_dir` should contain:
- `hex{hex_id}/` - Raw hexel data directory
- `df_fire_fru.csv` - Fire size distribution

If `prepare_data=True`, the pipeline will:

1. Build weather tables
2. Process fire size distributions
3. Split hexel into patches
4. Generate metadata CSV

Note: The data preparation has to be done at least once before running inference, as it creates the necessary datasets for the prediction loop. If you have already prepared the data, you can set `prepare_data=False` to skip this step in subsequent runs. If you add more hexels later, you can run with `prepare_data=True` with the `hex_id` set to the new hexel to prepare just that hexel's data.

## Folder Structure

```
inference/
├── data_dir/                                 # Hexel data directory (raw hexel data, fire size distribution)
├── outputs/                                  # Output directory for predictions, visualizations, and logs (appears after running)
├── predictor.py                              # Pure ML engine (tensor-in, tensor-out)
├── run_ai_surrogate_model_hexel_inference.py # Orchestration (data prep, dataset, prediction loop, post-processing)
├── config.yaml                               # Configuration file
└── run_hexel_inference.log                   # Output logs
```
## Components

The module is split into two components:

### 1. BurnRiskPredictor (`predictor.py`)
Pure inference class that wraps the PyTorch model. Handles only tensor operations - no data loading or hexel-specific logic.

### 2. run_ai_surrogate_model_hexel_inference (`run_ai_surrogate_model_hexel_inference.py`)
Orchestrates the full pipeline:
1. Data preparation (patch splitting, tabular data processing)
2. Dataset creation from checkpoint config
3. Prediction loop using `BurnRiskPredictor`
4. Post-processing and visualization
5. Saving results with log file output.


### Programmatic Usage

```python
from inference import BurnRiskPredictor, run_pipeline

# Option 1: Full end-to-end pipeline
predicted_hexel_grid, grid_profile = run_pipeline(
    checkpoint_path="experiments/best_model/best.pth",
    data_dir="data/",
    hex_id="02",
    prepare_data=True,
    save_dir="outputs/",  # Saves as predictions_hexel_02.npy
)

# Option 2: Just the predictor (for custom pipelines or serving)
predictor = BurnRiskPredictor.from_checkpoint(
    checkpoint_path="experiments/best_model/best.pth",
    spatial_channels=10,
    auxiliary_input_dims={"tabular_weather": 7, "tabular_fire_size": 5},
)
predicted_hexel_grid, grid_profile = predictor(spatial_batch, auxiliary_batch)
```

# Generating the full Canada map of hexels

To generate the full Canada hexel map of targets and/or predictions, run the following script (see --help for more args. information):

```python -m src.datasets.postprocessing.full_map.generate_full_hexel_map --data-dir data/ --scale "log" --show_hex_borders --output "experiments/full_canada_map.png"```

Note that to generate the full Canada maps, the `.tif` files of the hexels are required.

For example, to generate the full maps of ground truth targets (modify the `--data-dir` argument accordingly):

```python -m src.datasets.postprocessing.full_map.generate_full_hexel_map --data-dir <raw_data_folder/> --scale "log" --show_hex_borders --output "full_canada_map_targets.png"```

Similarly, to generate the full maps of obtained predictions from an AI surrogate model (modify the `--data-dir` argument accordingly):

```python -m src.datasets.postprocessing.full_map.generate_full_hexel_map --data-dir <model_outputs_folder/predicted_hexels> --pattern "*_predicted.tif" --scale "log" --show_hex_borders --output "full_canada_map_preds.png"```

You can also specify a fixed range of values for map generations via the `--vmin` and `--vmax` arguments.

Finally, to generate a map of residuals (preds - targets) instead (modify the `--target-dir` and `--pred-dir` arguments accordingly):

```python -m src.datasets.postprocessing.full_map.generate_full_hexel_diff_map --target-dir <raw_data_folder/> --target-pattern hex*/outputs/*_iter_bp.tif --pred-dir <model_outputs_folder/predicted_hexels> --pred-pattern "*_predicted.tif" --output "full_canada_map_diffs.png"```
