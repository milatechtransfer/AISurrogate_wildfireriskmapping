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

## Use the checkpoint on regional data

Use `inference/mechanistic_hybrid_v24.yaml` as the starting point. It points to the published checkpoint and the shared training artifacts, while `data_dir`, `hex_id`, and `save_dir` identify the regional study.

### Compatibility gate

The checkpoint is tied to the geometry and feature semantics used for training:

- Keep the regional rasters in their native projected CRS. **Do not reproject them to the national study CRS.**
- Every DEM, fuel, fire-zone, ignition, BP, FI, and ROS raster for one regional unit must already have the exact same CRS, affine transform, width, and height. The inference path uses `preserve_native_grid: true` and rejects a mismatch instead of resampling it.
- Cells must be square and exactly 100 m. V2.4's propagation physics has a fixed `100.0` m cell size.
- Rasters must be north-up, with no affine rotation or shear. Do not use bilinear or cubic interpolation on fuel or fire-zone codes.
- Supply at least 128 pixels, or 12.8 km, of raster context around the area for which predictions are needed. The model reads 512 x 512 context patches and predicts only their centered 256 x 256 regions.
- Check grid convergence across the regional extent. The national grids used for training were within approximately 0.6 degrees of true north, so no wind-axis correction was applied. If regional grid north differs materially from true north, rotate the wind components into raster-grid coordinates with a verified CRS convention before inference. Do not assume the existing conversion is sufficient.

The regional CRS may differ from the national CRS; the constraints are metric units, 100 m cells, north-up orientation, negligible or corrected grid convergence, and exact alignment among all rasters.

### Required layout

The pipeline remains hex-oriented. Represent each regional processing unit with a numeric ID such as `01`, even if it is not one of the national hexes:

```text
REGIONAL_ROOT/
├── df_fire_fru.csv
└── hex01/
    ├── spatial/
    │   ├── hex01_dem.tif
    │   ├── hex01_fbp.tif
    │   ├── hex01_firezones.tif
    │   ├── ignition_grids/
    │   │   └── hex01_ignGrid_<H-or-N>_<season>.tif
    │   └── mask_grids/
    │       ├── hex01_actual.shp
    │       └── associated shapefile sidecars
    ├── tabular/
    │   ├── hex01_DailyWeather.csv
    │   ├── hex01_FuelTypes.csv
    │   ├── hex01_FireZones.csv
    │   ├── hex01_GreenUp.csv
    │   ├── hex01_IgnitionDistribution.csv
    │   ├── hex01_IgnitionCount.csv
    │   └── hex01_ScenarioDistributions<...>FINAL.csv
    └── results/
        ├── burnP3Plus_OutputBurnProbability/
        │   └── burnProbability-sn2.tif
        ├── burnP3Plus_OutputFireIntensitySummaryMap/
        │   └── fbpSummary-FireIntensity-Average.tif
        └── burnP3Plus_OutputRateOfSpreadSummaryMap/
            └── fbpSummary-RateOfSpread-Average.tif
```

There must be exactly one `hex01_ScenarioDistributions*FINAL.csv`. Use equivalent names for another numeric ID.

**Current limitation:** the end-to-end inference preparation and plotting path uses the BP, FI, and ROS rasters as target channels, masks, and visual references. It therefore requires all three BurnP3+ result rasters even when the immediate goal is prediction. A feature-only regional study with no reference BP/FI/ROS rasters is not supported by this workflow; do not create zero-valued placeholder targets.

### Preserve the training feature contract

- **Ignition location:** provide the same Human (`H`) and Lightning (`N`) ignition-grid semantics and matching season names in `IgnitionDistribution.csv`. The pipeline combines them into two probability-mass channels whose joint spatial mass is scaled to 1,000,000. `FireZones.csv` must map fire-zone `Name` to integer `ID`. Inspect warnings about missing season, cause, or zone combinations; unresolved combinations are excluded.
- **Weather:** preserve the BurnP3+ columns and units used nationally. The pipeline computes `wind_x = WindSpeed * sin(WindDirection)` and `wind_y = WindSpeed * cos(WindDirection)` and retains the training convention that `WindDirection` is meteorological "from" direction. It reuses `weather_norm_params.json` from the shared training root; never fit normalization on the regional weather table.
- **Fire size:** `df_fire_fru.csv` must provide regional `GRIDCODE` and fire size in hectares. V2.4 spatializes q10, q50, and q90 of `LOG_SIZE_HA = log10(1 + hectares)`, adds the missing-zone mask, and reuses the national global-fill artifact. Do not introduce a regional min-max normalization.
- **Ignition count:** provide `hex##_IgnitionCount.csv` and its referenced scenario distribution. The pipeline derives log-mean and coefficient-of-variation features from the regional tables, even if the numeric ID overlaps a national hex, then applies the national ranges from `ignition_count_norm_params.json`. Values outside `[0, 1]` are intentionally not clipped and indicate conditions beyond the training range.
- **Fuel:** keep the Canadian FBP integer-code semantics. V2.4 uses the shared national iROS/HFI curve definitions, combined with the regional `GreenUp.csv` and ignition-season weights. Unknown codes fail explicitly. Adding a technically valid new curve does not make that fuel class in-distribution.
- **Elevation and outputs:** preserve metres for elevation and the same BP, FI, and ROS definitions and units as the national BurnP3+ products.

`training_data_root` in the supplied inference config is deliberate. The published checkpoint embeds the original training run's personal scratch path; the override leaves the checkpoint weights untouched while redirecting normalization files, fuel curves, and other training artifacts to their durable shared location. Regional raw tables are still used for regional season weights and ignition-count construction.

### Preflight checks

Before running a full region:

1. Open every raster together and confirm identical CRS, transform, width, and height, with 100 m square pixels.
2. Confirm the raster affine is north-up and assess grid convergence over the full study area.
3. Compare regional weather, elevation, fire-size, and ignition-count ranges with the national training ranges. Extreme normalized values are an out-of-distribution warning, not a preprocessing error to hide by clipping.
4. Confirm every raster fuel code has a corresponding shared curve and the expected FBP meaning.
5. Run one small regional unit first and inspect preprocessing warnings, prediction maps, seams, nodata boundaries, and the BP/FI/ROS value distributions.

### Run regional inference

From the repository root:

```bash
uv run python -m inference.run_ai_surrogate_model_hexel_inference \
  --config inference/mechanistic_hybrid_v24.yaml \
  --data_dir /path/to/REGIONAL_ROOT \
  --hex_id 01 \
  --save_dir /path/to/regional_v24_outputs
```

Use a new output directory when changing inputs. The first run prepares patches under `REGIONAL_ROOT/data_samples_approach_1/`; subsequent unchanged runs may pass `--prepare_data=False`.

The principal outputs are:

```text
/path/to/regional_v24_outputs/
├── predicted_patches/hexel_01.npy
├── predicted_hexels/hexel_01_bp_predicted.tif
├── predicted_hexels/hexel_01_fi_predicted.tif
├── predicted_hexels/hexel_01_ros_predicted.tif
├── predicted_hexels/hexel_01_hazard_predicted.tif
└── predicted_hexels_plot/
```

The hazard raster is the release workflow's raw BP/FI-derived hazard product. Keep the individual BP, FI, and ROS rasters for interpretation and quality control.

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
