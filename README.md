# 🔥 Accelerating Wildfire Hazard Mapping Across Canada with an AI Surrogate Modelling Framework

![python](https://img.shields.io/badge/python-3.12%2B-blue)
![package manager](https://img.shields.io/badge/package%20manager-uv-de5fe9)

An AI surrogate modelling framework that approximates wildfire burn probability, fire intensity, and rate of spread emulating process-based wildfire simulators (BurnP3 / BurnP3+).

## Table of contents

- [🔥 Accelerating Wildfire Hazard Mapping Across Canada with an AI Surrogate Modelling Framework](#-accelerating-wildfire-hazard-mapping-across-canada-with-an-ai-surrogate-modelling-framework)
  - [Table of contents](#table-of-contents)
  - [🛠️ Installation \& Setup](#️-installation--setup)
  - [📂 Data preparation](#-data-preparation)
  - [Trained model checkpoints](#trained-model-checkpoints)
  - [Model inference](#model-inference)
  - [Model Finetuning](#model-finetuning)
  - [Hazard evaluation](#hazard-evaluation)
  - [Counterfactual analysis](#counterfactual-analysis)
  - [🤖 Training](#-training)
    - [Logging with Comet](#logging-with-comet)
    - [Local training](#local-training)
    - [On the cluster](#on-the-cluster)
    - [Hazard evaluation on the cluster](#hazard-evaluation-on-the-cluster)
    - [⚙️ Runtime and memory profiling](#️-runtime-and-memory-profiling)
    - [Baselines](#baselines)

## 🛠️ Installation & Setup

Install `uv`: https://docs.astral.sh/uv/getting-started/installation.

The released checkpoints under `model_checkpoints/` (see [Trained model checkpoints](#trained-model-checkpoints)) are stored via [Git LFS](https://git-lfs.com). Install it **before** cloning, then run `git lfs install` once per machine:

```bash
git lfs install
```

Without this, `model_checkpoints/**/*.pth` files are checked out as small Git LFS pointer text files instead of the actual checkpoint weights, and `torch.load` will fail on them.

Clone the repository:

```bash
git clone https://github.com/milatechtransfer/nrcan_wildfireriskmapping.git
cd nrcan_wildfireriskmapping
```

If you already cloned the repository before installing Git LFS, fetch the real checkpoint contents with:

```bash
git lfs pull
```

Then, create/update the environment from the lockfile:

```bash
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

Add a new dependency/package to the codebase:

```bash
uv add <PACKAGE>
```

This automatically updates `pyproject.toml` to include the new package, regenerates `uv.lock`, and installs the package into `.venv`.

Activate the pre-commit hooks:

```bash
uv run pre-commit install
```

## 📂 Data preparation

For all data preparation steps, refer to [`data_preparation/README.md`](data_preparation/README.md).

## Trained model checkpoints

Under `model_checkpoints/` we release the best model checkpoints for the multi-output models (spatial-only, and spatial + weather, spatial + weather + fire size (TODO)). The latest checkpoint, `model_checkpoints/multi_task_spatial_weather_firesize/best.pth`, is the one used for all analysis throughout the paper, corresponding to [`configs/multi_output_spatial_weather_firesize_q3.yaml`](configs/multi_output_spatial_weather_firesize_q3.yaml).

These `.pth` files are stored via Git LFS (see [Installation & Setup](#️-installation--setup)). If `git lfs install` wasn't run before cloning, run `git lfs pull` to replace the LFS pointer files with the actual checkpoint weights before loading them.

## Model inference

To run the standalone inference pipeline, refer to [inference/README.md](inference/README.md)

## Model Finetuning

To prepare new regional dataset, refer to [`data_preparation/README_regional.md`](data_preparation/README.md).

Fine-tuning adapts an already-trained multi-output checkpoint (BP/FI/ROS) to a new, typically smaller, regional/scenario-specific dataset instead of training from scratch. The skeleton config [`configs/model_finetune.yaml`](configs/model_finetune.yaml) is a ready-to-edit template for this — fill in its `<PLACEHOLDER>` values (data paths, splits, experiment/run names, etc.) for your fine-tuning dataset. It is based on [`configs/multi_output_spatial_weather.yaml`](configs/multi_output_spatial_weather.yaml), with the same overrides used by the NWT example, [`configs/NWT_data/multi_output_NWT_eval_scenario_FireExcludeSpotting_finetune.yaml`](configs/NWT_data/multi_output_NWT_eval_scenario_FireExcludeSpotting_finetune.yaml):

- Sets `training.warm_start_checkpoint` to the base checkpoint, renamed `best_base.pth` and placed under the fine-tune run's own `save_dir`. Only `model_state` is loaded from this checkpoint — the optimizer, LR scheduler, epoch counter, and best-metric baseline all start fresh, so the run behaves like fine-tuning rather than resuming.
- Lowers the optimizer learning rate (`optimizer.lr: 5.0e-5` vs `7.0e-4` in the base config) and reduces `training.max_epochs` (`20` vs `25`) to avoid overfitting the much smaller fine-tuning dataset.
- Reduces `data.batch_size` (`16` vs `64`) to fit smaller fine-tuning datasets.
- Sets `evaluation.checkpoint_filename: "best.pth"` so the fine-tuned checkpoint is saved separately from `best_base.pth`.
- Optionally, `training.freeze_modules` (commented out by default, e.g. `["encoder", "bottleneck"]`) can freeze those submodules — excluding them from the optimizer and keeping them in `eval()` mode — so fine-tuning only updates the remaining layers (e.g. task heads).

To fine-tune:

1. Edit [`configs/model_finetune.yaml`](configs/model_finetune.yaml), filling in `save_dir`, `data.root_dir`/`raw_data_dir`/splits, and `logger`/experiment names for your fine-tuning dataset.
2. Copy/rename the base checkpoint to warm-start from (e.g. `model_checkpoints/multi_task_spatial_weather/best.pth`) to `<save_dir>/best_base.pth` (the path set in `training.warm_start_checkpoint`).
3. Run:

```bash
python -m src.train --config=configs/model_finetune.yaml
```

The fine-tuned checkpoint is written to `<save_dir>/best.pth`, and can be evaluated the same way as any other multi-output checkpoint (see [Model inference](#model-inference) and [Hazard evaluation](#hazard-evaluation)).

## Hazard evaluation

Hazard evaluation uses a single trained multi-output checkpoint that jointly predicts burn probability (BP) and fire intensity (FI) — optionally alongside other targets, e.g. ROS — over the same test regions, then computes:

- **Raw hazard**: `BP * min(FI, fi_cap)`
- **Scaled hazard**: `raw_hazard * scale_to / denominator`
- **Binned hazard classes** from the scaled hazard thresholds

To reproduce hazard results, provide a hazard config plus a compatible multi-output model config and checkpoint. The default example is `configs/hazard_eval_common_input_pipeline.yaml`, which points to an archived checkpoint-compatible multi-output config:

- `configs/archived/multi_output_common_input_pipeline_checkpoint.yaml`

The model config's `save_dir` and the hazard config's `checkpoint_filename` determine where the checkpoint is loaded from, e.g. `experiments/multi_output_common_input_pipeline/best.pth`. To evaluate a newer checkpoint, update the hazard config's `model.config_path` and `model.checkpoint_filename` as needed. The model's grid targets must include both `bp` and `fi`, and it is evaluated with the hazard config's shared `root_dir`, `raw_data_dir`, and `test_split`.

Run locally:

```bash
uv run python -m src.evaluate_hazard --config configs/hazard_eval_spatial_weather.yaml
```

The denominator policy controls how scaled and binned hazard are normalized:

| Policy | Meaning | Typical use |
| --- | --- | --- |
| `scale_denominator` | Use an explicit numeric denominator from the config. | Most reproducible when a fixed reference denominator is known. |
| `reference_file` | Read the denominator from `reference_denominator_path`. | Reusing a previously computed denominator. |
| `all_raw_ground_truth` | Compute the max raw hazard over all raw BP/FI rasters. | Default full-data reference for evaluation. |
| `train_ground_truth` | Compute the max raw hazard over train-split raw rasters. | Leak-safe model comparison. |
| `eval_ground_truth` | Compute the max raw hazard over the evaluated regions' ground truth. | Self-contained test-subset reports. |
| `prediction` | Compute the max raw hazard over model predictions. | Relative/model-dependent scaling when no reference exists. |

With `--self_normalized_prediction`, ground truth keeps the configured/reference denominator, while predictions are also scaled by the prediction max for diagnostic relative-hazard evaluation.

When `save_hazard_map: true`, hazard raster artifacts are written as a bundle for raw, scaled, and binned hazard. Use `--metrics_only` to skip raster/plot artifacts entirely, or `--skip_plots` to keep GeoTIFFs but skip per-region PNG plots. The default output directory is the hazard config's `save_dir`; key outputs include `hazard_scale_denominator.json`, `hazard_metrics_per_hex.csv`, `hazard_metrics_summary.json`, `hazard_confusion_matrix.csv`, `hazard_confusion_matrix.png`, and per-region GeoTIFFs under the legacy-named `hazard_hexels/` directory.

For SLURM/cluster instructions to run hazard evaluation (including multi-seed and buffer-only runs), see [Hazard evaluation on the cluster](#hazard-evaluation-on-the-cluster).

## Counterfactual analysis

Refer to the READMEs under [`src/datasets/postprocessing/counterfactual`](src/datasets/postprocessing/counterfactual).

## 🤖 Training

### Logging with Comet

To set up logging with Comet, add your API key:

```bash
export COMET_API_KEY=<YOUR_KEY>
```

### Local training

```bash
python -m src.train --config=configs/multi_output_spatial_weather.yaml
```

### On the cluster

To launch a job on the cluster, use the script `run_files/train.sh`.

1. `export COMET_API_KEY=YOUR_KEY`
2. Run `uv sync`, if needed.
3. Run with the desired config filename: `sbatch run_files/train.sh configs/multi_output_spatial_weather.yaml`. By default, it uses `configs/multi_output_spatial_weather.yaml`. All config files, including ablations, are available under `configs/`.

### Hazard evaluation on the cluster

Use the SLURM wrapper to run the hazard evaluation described in [Hazard evaluation](#hazard-evaluation):

```bash
sbatch run_files/eval_hazard.sh configs/hazard_eval_spatial_weather.yaml
```

For the three seeded 256x256 q10/q50/q90 fire-size checkpoints, run:

```bash
sbatch run_files/eval_hazard_multi_run.sh configs/hazard_eval_spatial_weather_firesize_q3.yaml
```

The array tasks use seeds `42`, `1337`, and `2024`, normalize each run by its maximum predicted raw hazard on the actual support, and write results under `experiments/hazard_eval_multi_output_spatial_weather_firesize_q3/seed_<seed>/`.

Extra CLI arguments can be passed through `EVAL_ARGS`. For example, to run a buffer-only evaluation from a different data root and keep outputs separate:

```bash
EVAL_ARGS="--mask_scope buffer_only --root_dir /network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v2_buffer_test --save_dir experiments/hazard_eval_common_input_pipeline_buffer_only --skip_plots" \
  sbatch run_files/eval_hazard.sh configs/hazard_eval_common_input_pipeline.yaml
```

### ⚙️ Runtime and memory profiling

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

Swap `bp_` for `fi_`/`ros_` to target fire intensity or rate of spread. The mean-value baseline is deterministic and only needs a single seed.
