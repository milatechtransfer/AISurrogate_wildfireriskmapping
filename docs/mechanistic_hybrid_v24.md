# Mechanistic hybrid v2.4

Mechanistic hybrid v2.4 is a native-grid CNN-physics model that jointly predicts burn probability (BP), fire intensity (FI), and rate of spread (ROS). This release provides a verified reference checkpoint, a standalone training config, shared prepared data, and direct Slurm jobs for smoke testing, training, and stitched evaluation.

## Reference artifacts

| Artifact | Location |
| --- | --- |
| Release branch | `release/mechanistic-hybrid-v24` |
| Config | `configs/mechanistic_hybrid_v24_reference.yaml` |
| Prepared data | `/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v6_native_context_512_crop_256_ignition_probability_mass_log_firesize` |
| Checkpoint bundle | `/network/projects/amlrt/nrcan_wildfires/checkpoints/burnp3plus/final_experiments/mechanistic_hybrid_v24_native_512_crop_256_firesize_q3` |
| Checkpoint SHA-256 | `ecc5836e2cfd736e8fb5133115f55520ca21ab15a629b952a9f33f69ba94c0f3` |

The config is standalone: it has no `extends` chain and already points to the shared dataset. Comet logging is disabled by default.

## Setup

```bash
git switch release/mechanistic-hybrid-v24
uv sync
mkdir -p logs
```

## Smoke test

```bash
sbatch run_files/mechanistic_hybrid_v24/smoke.sh
```

This loads a real shared-data batch and runs one forward, backward, and optimizer step on an A100.

## Train

```bash
sbatch run_files/mechanistic_hybrid_v24/train.sh
```

The reference recipe uses native 100 m grids, 512-pixel context patches, centered 256-pixel targets, batch size 4, gradient accumulation 16, and 40 epochs. Outputs are written to:

```text
experiments/mechanistic/mechanistic_hybrid_v24_native_512_crop_256_firesize_q3/
```

Comet is disabled in the reference config. To enable it, copy the config, set `logger.enabled: true`, export `COMET_API_KEY`, and submit with `CONFIG`:

```bash
CONFIG=configs/my_v24_experiment.yaml sbatch --export=ALL run_files/mechanistic_hybrid_v24/train.sh
```

The reference run used one A100 40 GB, took `1-14:45:13`, and reached approximately 14.3 GB maximum process GPU memory.

## Evaluate

```bash
sbatch run_files/mechanistic_hybrid_v24/evaluate.sh
```

The default job evaluates all five test hexes and writes stitched per-hex metrics, prediction GeoTIFFs, diagnostic plots, and `test_metrics.csv` to:

```text
experiments/mechanistic/mechanistic_hybrid_v24_reference_evaluation/
```

Use another checkpoint or output directory with environment variables:

```bash
CHECKPOINT=/path/to/best.pth OUTPUT_DIR=experiments/my_v24_eval \
  sbatch --export=ALL run_files/mechanistic_hybrid_v24/evaluate.sh
```

For a metrics-only run without rasters or plots:

```bash
EVAL_ARGS="--metrics_only" sbatch --export=ALL run_files/mechanistic_hybrid_v24/evaluate.sh
```

To keep prediction rasters but skip only the plots:

```bash
EVAL_ARGS="--skip_hexel_plots" sbatch --export=ALL run_files/mechanistic_hybrid_v24/evaluate.sh
```

The reference five-hex evaluation took `00:06:26` on an A100.

## Reference performance

| Split | BP CCC | FI CCC | ROS CCC | Mean CCC |
| --- | ---: | ---: | ---: | ---: |
| Validation, selected checkpoint | 0.8256 | 0.7617 | 0.7298 | 0.772381 |
| Stitched test, five hexes | 0.819621 | 0.782212 | 0.711031 | 0.770954 |

The stitched test top-10 IoU is `0.520382` for BP, `0.336043` for FI, and `0.286883` for ROS. These are single-seed reference results.

## Adapt the model to another task

Copy `configs/mechanistic_hybrid_v24_reference.yaml`, then give the experiment a new `save_dir` and `logger.experiment_name`. A compatible prepared dataset must preserve the same input contract:

- native, aligned 100 m grids;
- human and lightning ignition probability mass;
- elevation and raw fuel grids;
- spatialized ISI and wind components;
- q10/q50/q90 direct log-hectare fire-size channels plus the missing-zone mask;
- normalized ignition-count mean and coefficient of variation;
- iROS and HFI fuel curves;
- BP, FI, and ROS targets.

If the input contract or targets change, update the config and add a focused smoke test before launching full training.
