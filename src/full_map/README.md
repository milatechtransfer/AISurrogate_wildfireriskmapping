# Full-Canada map generation

`src/full_map/` generates a full-Canada wildfire-risk vis: it runs model inference over
**every** hexel (train + val + test combined), saves per-hexel predicted rasters, then
mosaics those rasters onto the real national grid (using the true hexel geometries) and
compares the mosaic against an already-stitched national ground-truth raster. The model
predicts three targets — `bp` (burn probability), `fi` (fire intensity), `ros` (rate of
spread) — so predictions, mosaics, and diffs are all produced **per target**.

Pipeline (each step reads its config via `--config`):

1. `generate_predictions.py` — inference + per-hexel `.tif` predictions + per-split metrics CSV
2. `generate_full_hexel_map.py` — mosaics predicted hexels into one national raster per target
3. `generate_full_hexel_diff_map.py` — diffs each national mosaic against its GT raster

## Config

Add a `full_map:` section to your YAML config:

```yaml
full_map:
  national_shapefile_path: /path/to/national_hexel_polygons.shp
  hexel_id_column: hex_id  # default; column in the shapefile holding each polygon's hex_id
  national_gt_raster_paths:
    bp: /path/to/national_bp_ground_truth.tif
    fi: /path/to/national_fi_ground_truth.tif
    ros: /path/to/national_ros_ground_truth.tif
```

`national_shapefile_path` and `national_gt_raster_paths` are only required for steps 2 and 3
(mosaicking/diffing); step 1 (prediction generation) does not need them.

## 1. Generate predictions for all hexels

```
python -m src.full_map.generate_predictions --config path/to/config.yaml
```

- Runs inference over `train`, `val`, and `test` splits (no shuffling), loading the checkpoint
  at `config.evaluation.checkpoint_filename`.
- Comet logging is always disabled and no diagnostic plots are saved, regardless of what the
  config says.
- For each split, writes predicted hexel rasters to
  `config.save_dir/<split>/predicted_hexels/hexel_{hex_id}_{target}_predicted.tif` (one `.tif`
  per hexel per target — 3 per hexel for `bp`/`fi`/`ros`).
- Writes one wide-format `config.save_dir/<split>/hexel_metrics.csv` per split (one row per
  hexel, metric names as columns).

Useful flags: `--stitch_mode {mean,max}` (default `mean`), `--mask_scope`,
`--report_firezone_metrics`, `--run_id` (applies the same seed/save_dir overrides used by
training's SLURM array jobs). See `--help` for details.

This step runs a full forward pass over every hexel and needs a GPU, so on Mila/SLURM clusters
submit it as a job rather than running it on the login node:

```bash
sbatch run_files/full_map/generate_predictions.sh configs/your_config.yaml
```

Pass extra CLI flags via `PRED_ARGS`, e.g.:

```bash
PRED_ARGS="--stitch_mode=max --report_firezone_metrics" sbatch run_files/full_map/generate_predictions.sh configs/your_config.yaml
```

## 2. Mosaic predictions onto the national grid

```
python -m src.full_map.generate_full_hexel_map --config path/to/config.yaml \
    --output-dir experiments/full_map
```

- Reads `config.full_map.national_shapefile_path` / `hexel_id_column` to look up each hexel's
  real polygon, and reprojects each predicted hexel raster onto the grid (CRS/transform/shape)
  of that target's `config.full_map.national_gt_raster_paths[target]`, pasting valid pixels
  into a national canvas.
- Defaults to reading predicted hexels from `config.save_dir` (override with `--pred-root`),
  matching the glob `--pattern` (default `*/predicted_hexels/hexel_*_predicted.tif`, i.e. all
  splits combined).
- Writes one `{target}_national_predicted_map.tif` per target (with a configured GT raster)
  into `--output-dir`. Pass `--save-plots` to also save a `{target}_national_predicted_map.png`
  (`--scale {linear,log}` controls color scaling, `--title` sets the plot title prefix).
- Each predicted hexel is reprojected only into the small destination window covering its own
  footprint (not the full national canvas), so memory/compute scale with the number and size
  of hexels rather than with the national raster size per hexel.
- Pass `--skip-existing` to skip (re)building a target's mosaic if its
  `{target}_national_predicted_map.tif` already exists in `--output-dir` -- useful for resuming
  after a job was killed partway through the target loop, without redoing already-finished
  targets.

This step is CPU-only (no GPU needed) but can still be memory-hungry for a Canada-wide
reference raster (one full national float32 array is held in memory per target). Submit it
as a job rather than running it on the login node:

```bash
sbatch run_files/full_map/generate_full_hexel_map.sh configs/your_config.yaml
```

Override `PRED_ROOT`, `OUTPUT_DIR`, or pass extra flags via `MOSAIC_ARGS`, e.g.:

```bash
MOSAIC_ARGS="--save-plots --scale=log" sbatch run_files/full_map/generate_full_hexel_map.sh configs/your_config.yaml
```

Adjust the script's `--mem` to comfortably fit `height * width * 4 bytes` for your national
reference raster (check with
`python -c "import rasterio; s = rasterio.open('<path>'); print(s.height, s.width, s.height*s.width*4/1e9, 'GB')"`).
At 100m resolution and a full Canada-wide extent (~55,000 x 46,000 px), that's ~10GB, so the
default 48Gb has comfortable headroom; a smaller/regional reference raster needs much less.

## 3. Diff mosaics against ground truth

```
python -m src.full_map.generate_full_hexel_diff_map --config path/to/config.yaml \
    --mosaic-dir experiments/full_map --output-dir experiments/full_map
```

- For each target in `config.full_map.national_gt_raster_paths`, loads
  `{target}_national_predicted_map.tif` from `--mosaic-dir` (produced by step 2) and computes
  `prediction - ground_truth` on the shared grid (targets whose mosaic file is missing are
  skipped with a warning, not a hard failure).
- Writes `{target}_national_diff_map.tif` per target into `--output-dir`, and prints/returns
  summary metrics (`ccc`, `spearman`, `normalized_mae`, `n_valid_pixels`) computed over
  pixels valid in both rasters. Pass `--save-plots` for a `{target}_national_diff_map.png`
  (red/blue diverging colormap centered at 0).
- Pass `--skip-existing` to skip (re)computing a target's diff if its
  `{target}_national_diff_map.tif` already exists in `--output-dir` (e.g. to resume after a
  killed job). Skipped targets are omitted from the returned metrics dict.

This step is also CPU-only but loads two full national rasters into memory per target; submit
via:

```bash
sbatch run_files/full_map/generate_full_hexel_diff_map.sh configs/your_config.yaml
```

## Module layout

- `generate_predictions.py` — inference + per-hexel raster/metric export (step 1)
- `generate_full_hexel_map.py` — `generate_national_mosaics(...)`, real-CRS mosaicking (step 2)
- `generate_full_hexel_diff_map.py` — `generate_national_diffs(...)` / `compute_national_diff`, GT comparison (step 3)
- `utils.py` — shared helpers: `load_hexel_shapefile`, `group_predicted_hexel_files_by_target`,
  `mosaic_predicted_hexels`, `calculate_global_stats`, `get_scale_settings`

See `tests/test_full_map.py` for runnable examples of each helper against tiny synthetic
rasters/shapefiles.
