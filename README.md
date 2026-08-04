## Canada-Wide WildFire Risk Mapping

### Installation & Setup

Install `uv`: https://docs.astral.sh/uv/getting-started/installation.

Clone the repository:
```bash
   git clone https://github.com/milatechtransfer/nrcan_wildfireriskmapping.git
   cd nrcan_wildfireriskmapping
```
Then, create/update the environment from the lockfile:
```bash
   uv sync
```
To activate the environment:
```bash
source .venv/bin/activate
```
To add a new dependency/package into the codebase:
```bash
uv add <PACKAGE>
```
This will automatically update the `pyproject.toml` to include the new package, as well as regenerate the updated `uv.lock` and install the new package into the `.venv`.

To activate the pre-commit hooks, run:
```bash
uv run pre-commit install
```

### Logging with Comet

To set up the logging with Comet, add your API key via:
```bash
export COMET_API_KEY=<YOUR_KEY>
```

### Data preparation

For all the data preparation steps, refer to [the following section](data_preparation/README.md).

### Training

```python -m src.train --config=configs/default_v1.yaml```

#### On the cluster:
To launch a job on the cluster, use the script `run_files/train.sh`.
Steps:
1. `export COMET_API_KEY=YOUR_KEY`
2. Run `uv sync`, if needed
3. Run with the desired config filename `sbatch run_files/train.sh configs/default_v1_full_data.yaml`. By default, it uses `configs/default_v1.yaml`.

### Inference

To visualize predictions and/or save visualizations, add the optional flags `--visualize_predictions` and/or `--save_visualizations`, respectively.

```python -m src.evaluate_hexels --config=configs/default_v1.yaml```

### Hazard evaluation

Hazard evaluation combines a trained burn probability (BP) checkpoint and a trained fire intensity (FI) checkpoint over the same test hexels, then computes:

- raw hazard: `BP * min(FI, fi_cap)`
- scaled hazard: `raw_hazard * scale_to / denominator`
- binned hazard classes from the scaled hazard thresholds

To reproduce hazard results, provide a hazard config plus compatible BP/FI model configs and checkpoints. The default example is `configs/hazard_eval_common_input_pipeline.yaml`, which points to archived checkpoint-compatible BP/FI configs:

- `configs/archived/bp_common_input_pipeline_checkpoint.yaml`
- `configs/archived/fi_common_input_pipeline_checkpoint.yaml`

Each model config's `save_dir` and the hazard config's `checkpoint_filename` determine where the checkpoint is loaded from, e.g. `experiments/bp_common_input_pipeline/best.pth` and `experiments/fi_common_input_pipeline/best.pth`. To evaluate newer checkpoints, update the hazard config's `bp.config_path`, `fi.config_path`, and checkpoint filenames as needed. The BP config must target `bp`, the FI config must target `fi`, and both are evaluated with the hazard config's shared `root_dir`, `raw_data_dir`, and `test_split`.

Run locally with:

```bash
uv run python -m src.evaluate_hazard --config configs/hazard_eval_common_input_pipeline.yaml
```

On the cluster, use the SLURM wrapper:

```bash
sbatch run_files/eval_hazard.sh configs/hazard_eval_common_input_pipeline.yaml
```

Extra CLI arguments can be passed through `EVAL_ARGS`. For example, to run a buffer-only evaluation from a different data root and keep outputs separate:

```bash
EVAL_ARGS="--mask_scope buffer_only --root_dir /network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v2_buffer_test --save_dir experiments/hazard_eval_common_input_pipeline_buffer_only --skip_plots" \
  sbatch run_files/eval_hazard.sh configs/hazard_eval_common_input_pipeline.yaml
```

The denominator policy controls how scaled and binned hazard are normalized:

| Policy | Meaning | Typical use |
| --- | --- | --- |
| `scale_denominator` | Use an explicit numeric denominator from the config. | Most reproducible when a fixed reference denominator is known. |
| `reference_file` | Read the denominator from `reference_denominator_path`. | Reusing a previously computed denominator. |
| `all_raw_ground_truth` | Compute the max raw hazard over all raw BP/FI rasters. | Default full-data reference for evaluation. |
| `train_ground_truth` | Compute the max raw hazard over train-split raw rasters. | Leak-safe model comparison. |
| `eval_ground_truth` | Compute the max raw hazard over the evaluated hexels' ground truth. | Self-contained test-subset reports. |
| `prediction` | Compute the max raw hazard over model predictions. | Relative/model-dependent scaling when no reference exists. |

With `--self_normalized_prediction`, ground truth keeps the configured/reference denominator, while predictions are also scaled by the prediction max for diagnostic relative-hazard evaluation.

When `save_hazard_map: true`, hazard raster artifacts are written as a bundle for raw, scaled, and binned hazard. Use `--metrics_only` to skip raster/plot artifacts entirely, or `--skip_plots` to keep GeoTIFFs but skip per-hexel PNG plots. The default output directory is the hazard config's `save_dir`; key outputs include `hazard_scale_denominator.json`, `hazard_metrics_per_hex.csv`, `hazard_metrics_summary.json`, `hazard_confusion_matrix.csv`, `hazard_confusion_matrix.png`, and per-hexel GeoTIFFs under `hazard_hexels/`.

### Generate full Canada map of hexels

To generate the full Canada hexel map of targets and/or predictions, run the following script (see --help for more args. information):

```python -m src.datasets.postprocessing.full_map.generate_full_hexel_map --data-dir data/ --scale "log" --show_hex_borders --output "experiments/full_canada_map.png"```

Note that to generate full Canada maps, the `.tif` files of the hexels are required.

For example, to generate the full maps of ground truth targets on the cluster:

```python -m src.datasets.postprocessing.full_map.generate_full_hexel_map --data-dir /network/projects/amlrt/nrcan_wildfires/full_data/yan_bp3/ --scale "log" --show_hex_borders --output "experiments/full_canada_map_targets.png"```

Similarly, to generate the full maps of obtained predictions from an AI surrogate model on the cluster:

```python -m src.datasets.postprocessing.full_map.generate_full_hexel_map --data-dir experiments/final_model_outputs/predicted_hexels/ --pattern "*_predicted.tif" --scale "log" --show_hex_borders --output "experiments/full_canada_map_preds.png"```

You can also specify a fixed range of values for map generations via the `--vmin` and `--vmax` arguments.

Finally, to generate a map of residuals (preds - targets) on the cluster:

```python -m src.datasets.postprocessing.full_map.generate_full_hexel_diff_map --target-dir /network/projects/amlrt/nrcan_wildfires/full_data/yan_bp3/ --target-pattern hex*/outputs/*_iter_bp.tif --pred-dir experiments/unet_full_data_spatial_weather_new_config/predicted_hexels/ --pred-pattern "*_predicted.tif" --output "experiments/full_canada_map_diffs.png"```

### Baselines

Tabular and reference baselines (XGBoost, mean-value) share the entrypoint `src/train_tabular_baseline.py`, with configs in `configs/baselines/` and run scripts in `run_files/baselines/`.

```bash
sbatch run_files/baselines/mean_baseline.sh configs/baselines/bp_mean_baseline.yaml
sbatch run_files/baselines/train_baseline.sh configs/baselines/bp_spatial_only_xgb.yaml
```

For multi-seed XGBoost runs (array job, seeds 0-2 by default):

```bash
sbatch run_files/baselines/train_baseline_multi_run.sh configs/baselines/bp_spatial_only_xgb.yaml
```

Swap `bp_` for `fi_`/`ros_` to target fire intensity or rate of spread. Mean-value baseline is deterministic and only needs a single seed.
