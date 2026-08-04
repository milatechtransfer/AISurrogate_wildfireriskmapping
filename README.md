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

```python -m src.train --config=configs/multi_output_spatial_weather.yaml```

#### On the cluster:
To launch a job on the cluster, use the script `run_files/train.sh`.
Steps:
1. `export COMET_API_KEY=YOUR_KEY`
2. Run `uv sync`, if needed
3. Run with the desired config filename `sbatch run_files/train.sh configs/multi_output_spatial_weather.yaml`. By default, it uses `configs/multi_output_spatial_weather.yaml`. All config files as well as ablations are available udner `configs/`

### Inference

To visualize predictions and/or save visualizations, add the optional flags `--visualize_predictions` and/or `--save_visualizations`, respectively.

```python -m src.evaluate_hexels --config=configs/multi_output_spatial_weather.yaml```

### Hazard evaluation

Hazard evaluation uses a single trained multi-output checkpoint that jointly predicts burn probability (BP) and fire intensity (FI) (optionally alongside other targets, e.g. ROS) over the same test hexels, then computes:

- raw hazard: `BP * min(FI, fi_cap)`
- scaled hazard: `raw_hazard * scale_to / denominator`
- binned hazard classes from the scaled hazard thresholds

To reproduce hazard results, provide a hazard config plus a compatible multi-output model config and checkpoint. The default example is `configs/hazard_eval_common_input_pipeline.yaml`, which points to an archived checkpoint-compatible multi-output config:

- `configs/archived/multi_output_common_input_pipeline_checkpoint.yaml`

The model config's `save_dir` and the hazard config's `checkpoint_filename` determine where the checkpoint is loaded from, e.g. `experiments/multi_output_common_input_pipeline/best.pth`. To evaluate a newer checkpoint, update the hazard config's `model.config_path` and `model.checkpoint_filename` as needed. The model's grid targets must include both `bp` and `fi`, and it is evaluated with the hazard config's shared `root_dir`, `raw_data_dir`, and `test_split`.

Run locally with:

```bash
uv run python -m src.evaluate_hazard --config configs/hazard_eval_spatial_weather.yaml
```

On the cluster, use the SLURM wrapper:

```bash
sbatch run_files/eval_hazard.sh configs/hazard_eval_spatial_weather.yaml
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

### Counterfactual analysis
Refer to the READMEs under `src/datasets/postprocessing/counterfactual`

### Runtime and memory profiling

Benchmarks used to produce the runtime/memory comparison against BurnP3+ (paper Table: Time and Memory Performance). Configs live in `configs/runtime_benchmarks/`, scripts in `run_files/runtime_profiling/`.

Copy the checkpoint's `best.pth` to the `save_dir` referenced by the config, and make sure `data.root_dir` is reachable and contains `dataset_norm_stats.json` (without it, dataloader setup reverts from ~7s to ~70-90s).

```bash
# GPU
sbatch --partition=long --gres=gpu:l40s:1 --cpus-per-task=4 --time=00:30:00 \
  --export=N_REPS=10,EVAL_ARGS="--tif_only" \
  run_files/runtime_profiling/eval_hexel_benchmark_gpu.sh configs/runtime_benchmarks/<config>.yaml

# CPU
sbatch --partition=long-cpu --nodelist=<node> --cpus-per-task=4 --time=02:00:00 \
  --export=N_REPS=10,EVAL_ARGS="--tif_only" \
  run_files/runtime_profiling/eval_hexel_benchmark_cpu.sh configs/runtime_benchmarks/<config>.yaml

# MacBook (MPS), run locally
caffeinate -i env PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 N_REPS=10 EVAL_ARGS="--tif_only" \
  ./run_files/runtime_profiling/eval_hexel_benchmark_mac.sh configs/runtime_benchmarks/<config>_mac.yaml
```

`--tif_only` writes predicted rasters and skips metrics; use it for runtime comparisons since it matches BurnP3+'s actual output. `--fast_eval` computes metrics instead of writing rasters — check the resulting `hex16/bp_ccc` etc. against the checkpoint directory's `test_metrics.csv` before trusting timing from that mode.

Memory is reported differently per platform: `Peak GPU Reserved` for CUDA (allocator-reserved, the provisioning number), `Peak MPS Driver Allocated` for MPS (this is unified memory and already includes host RAM — don't add process RSS to it), `Peak Host RSS` for CPU.
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
