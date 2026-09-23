# 🔥 Wildfire AI Surrogate: Inference

![python](https://img.shields.io/badge/python-3.12%2B-blue)
![package manager](https://img.shields.io/badge/package%20manager-uv-de5fe9)
![hardware](https://img.shields.io/badge/runs%20on-CPU%20%7C%20CUDA%20%7C%20Apple%20MPS-green)
![os](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

Produce **burn probability (BP)**, **fire intensity (FI)**, **rate of spread (ROS)** and **wildfire hazard**
maps for a BurnP3+ project in minutes, using the AI surrogate instead of running BurnP3+ simulations.
Where BurnP3+ has already been run, compare the AI surrogate against it with one command.

You only need the model **bundle** (one folder) and your BurnP3+ project **inputs**. No training data,
GPU or BurnP3+ outputs are required to predict.

## Table of contents

- [🔥 Wildfire AI Surrogate: Inference](#-wildfire-ai-surrogate-inference)
  - [Table of contents](#table-of-contents)
  - [🛠️ Installation](#️-installation)
  - [🚀 Quick start](#-quick-start)
  - [📦 The model bundle](#-the-model-bundle)
  - [📂 Preparing your project](#-preparing-your-project)
    - [Fuel codes](#fuel-codes)
    - [Fire-size table](#fire-size-table)
  - [✅ Check your inputs](#-check-your-inputs)
  - [🗺️ Predict](#️-predict)
  - [📊 Evaluate against BurnP3+](#-evaluate-against-burnp3)
    - [Metrics](#metrics)
  - [🌲 Regional studies](#-regional-studies)
  - [🐍 Using it from Python](#-using-it-from-python)
  - [⚙️ Performance](#️-performance)
  - [🩺 Troubleshooting](#-troubleshooting)
  - [🔧 For maintainers: exporting a bundle](#-for-maintainers-exporting-a-bundle)

## 🛠️ Installation

Install `uv`: https://docs.astral.sh/uv/getting-started/installation.

Clone the repository and create the environment from the lockfile:

```bash
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

All commands below are run from the repository root. Instead of activating, you can also prefix them with
`uv run`, e.g. `uv run python -m inference.predict --help`.

## 🚀 Quick start

```bash
# 1. Check that the project has everything the model needs (seconds per region)
python -m inference.check --bundle nrcan-surrogate-bp-fi-ros-v1.0 --project path/to/project

# 2. Predict BP, FI, ROS and hazard for every region
python -m inference.predict --bundle nrcan-surrogate-bp-fi-ros-v1.0 --project path/to/project \
    --output path/to/predictions

# 3. (Optional) Where BurnP3+ outputs exist, measure how close the AI surrogate is to BurnP3+
python -m inference.evaluate --bundle nrcan-surrogate-bp-fi-ros-v1.0 --project path/to/project \
    --output path/to/evaluation
```

Every command has `--help`. The device (CUDA GPU, Apple MPS or CPU) is picked automatically; force one
with `--device cpu`.

## 📦 The model bundle

The AI surrogate is shipped as one folder, e.g. `nrcan-surrogate-bp-fi-ros-v1.0/`. Download
`nrcan-surrogate-bp-fi-ros-v1.0.zip` from the repository's
[Releases](https://github.com/milatechtransfer/nrcan_wildfireriskmapping/releases) page, unzip it, and point
`--bundle` at the folder.

| File | Content |
|---|---|
| `MODEL_CARD.md` | What the model predicts, its accuracy and its limitations. **Read this first.** |
| `model.pt` | The trained model |
| `manifest.yaml`, `norm/`, `lookups/`, `SHA256SUMS` | Settings and tables the model needs; checked automatically at start-up |

Do not edit files inside the bundle; pass your own tables with command-line options instead.

## 📂 Preparing your project

A project is a folder of one or more BurnP3+ regions, laid out as BurnP3+ writes them. Each region is a
folder named `hexNN` (in the national study, one per hexel):

```text
path/to/project/
└── hex12/
    ├── spatial/
    │   ├── hex12_dem.tif                     # elevation, ~100 m cells
    │   ├── hex12_fbp.tif                     # FBP fuel codes (or hex12_fbp_<scenario>.tif)
    │   ├── hex12_firezones.tif               # fire-zone IDs
    │   ├── ignition_grids/
    │   │   └── hex12_ignGrid_{H|N}_<season>.tif   # human / lightning ignition, per season
    │   └── mask_grids/
    │       └── hex12_actual.shp              # area to predict (optional, see Regional studies)
    ├── tabular/
    │   ├── hex12_DailyWeather.csv
    │   ├── hex12_FireZones.csv
    │   ├── hex12_IgnitionDistribution.csv
    │   ├── hex12_GreenUp.csv
    │   ├── hex12_FuelTypes.csv               # optional, see Fuel codes
    │   └── hex12_FuelCodeCrosswalk.csv       # optional, see Fuel codes
    └── results/                              # BurnP3+ outputs: only needed by evaluate
```

Rasters may be in any projected CRS; they are reprojected automatically.

### Fuel codes

The model knows every fuel code of the national BurnP3+ data. If your fuel grid uses other codes (e.g. new
mixedwood percentages in a regional grid), add the BurnP3+ fuel tables to `tabular/`:

- `hexNN_FuelTypes.csv` with columns `Name`, `ID`
- `hexNN_FuelCodeCrosswalk.csv` with columns `FuelType`, `Code` (e.g. `M-1/M-2 (25 PC)`)

New codes of these fuel types are then handled automatically: **C-1 to C-5, C-7, D-1/D-2, M-1/M-2 (any
percent conifer), O-1a/O-1b and non-fuel**. **C-6, M-3/M-4 and S-1 to S-3** are not supported: recode those
cells or set them to nodata.

### Fire-size table

A national fire-size table is included in the bundle and used by default. To use your own (e.g. regional
fire sizes), pass `--fire_size_table my_fires.csv` with one row per fire and columns `GRIDCODE` (fire-zone ID)
and `SIZE_HA` (hectares); BurnP3+'s `FRU` and `Fsize` column names also work.

## ✅ Check your inputs

`inference.check` reads the project and reports every problem at once, without running the model.
`predict` and `evaluate` run the same check first and stop on errors.

```text
$ python -m inference.check --bundle nrcan-surrogate-bp-fi-ros-v1.0 --project NWT_data \
      --scenario_name FireExcludeSpotting --mask_scope none

Checked 1 hexel(s) in NWT_data for model nrcan-surrogate-bp-fi-ros v1.0.0 (mask: none)

Project
  NOTE     Fire sizes: using the national training table shipped with the model (...)
hex100: OK
  NOTE     hex100/tabular/hex100_FuelCodeCrosswalk.csv: Fuel code(s) not in the model's fuel table, ...:
           425 = M-1 (25 PC) (161,615 cells), 505 = M-2 (05 PC) (170,755 cells), ...

Result: ready to predict, no problems found.
```

| Level | Meaning | Examples |
|---|---|---|
| 🔴 **ERROR** | Prediction would fail or be meaningless; nothing is run | Missing files, raster without a CRS, DEM not ~100 m, undefined or unsupported fuel codes, no weather for the region's fire zones |
| 🟡 **WARNING** | Prediction runs, but some cells use a fallback; review these | Fire zones without weather (region average used), zones missing from the fire-size table, implausible weather values |
| ⚪ **NOTE** | Information | Which fire-size table is used, new fuel codes found in the fuel tables |

Options: `--hex_ids 12 14`, `--scenario_name NAME`, `--mask_scope`, `--fire_size_table`, `--outputs` (also check
the BurnP3+ results used by `evaluate`) and `--json report.json`. Exit code `0` = ready, `1` = input errors,
`2` = the check could not run.

## 🗺️ Predict

```bash
python -m inference.predict --bundle nrcan-surrogate-bp-fi-ros-v1.0 --project path/to/project \
    --output path/to/predictions
```

Outputs, one folder per region (GeoTIFF, nodata `-9999`):

| File | Content |
|---|---|
| `hexNN/hexNN_bp.tif` | Burn probability (0–1) |
| `hexNN/hexNN_fi.tif` | Fire intensity (kW/m) |
| `hexNN/hexNN_ros.tif` | Rate of spread (m/min) |
| `hexNN/hexNN_hazard_raw.tif` | Hazard = BP × min(FI, 10 000 kW/m) |
| `hexNN/hexNN_hazard_scaled.tif` | Hazard on a 0–100 scale, comparable across studies |
| `hexNN/hexNN_hazard_class.tif` | 13 hazard classes (1 = lowest; 0 = nodata) |
| `run_manifest.json` | Record of the run (model, inputs, options), for reproducibility |
| `predict.log` | Run log |

Outputs are in Canada Lambert Conformal Conic (`ESRI:102002`, ~100 m cells). If your inputs are in another
CRS, reproject the outputs before overlaying them cell by cell.

| Option | Use |
|---|---|
| `--hex_ids 12 14` | Only these regions (`12` or `hex12`) |
| `--scenario_name NAME` | Use `hexNN_fbp_NAME.tif` as the fuel grid |
| `--mask_scope actual\|buffer\|none` | Area to predict: region mask (default), buffered mask, or the whole raster |
| `--fire_size_table FILE` | Your own fire sizes instead of the national table |
| `--device auto\|cpu\|cuda\|mps` | Hardware to run on |
| `--batch_size N` | Lower it if memory runs out |
| `--no_hazard` | Skip the hazard rasters |
| `--overwrite` | Replace earlier predictions in `--output` |

## 📊 Evaluate against BurnP3+

Where BurnP3+ has been run, `inference.evaluate` predicts (or reuses predictions) and scores the AI surrogate
against the BurnP3+ outputs in each region's `results/` folder:

- national layout: `results/burnP3Plus_OutputBurnProbability/burnProbability-sn2.tif`,
  `results/burnP3Plus_OutputFireIntensitySummaryMap/fbpSummary-FireIntensity-Average.tif` and
  `results/burnP3Plus_OutputRateOfSpreadSummaryMap/fbpSummary-RateOfSpread-Average.tif`;
- with `--scenario_name NAME`: `results/NAME/burnP3Plus_Output..._NAME_{All|Average}.tif`.

```bash
# Predict and evaluate (predictions are kept in path/to/evaluation/predictions/)
python -m inference.evaluate --bundle nrcan-surrogate-bp-fi-ros-v1.0 --project path/to/project \
    --output path/to/evaluation

# Score predictions made earlier with inference.predict (the model is not run again)
python -m inference.evaluate --bundle nrcan-surrogate-bp-fi-ros-v1.0 --project path/to/project \
    --predictions path/to/predictions --output path/to/evaluation
```

A summary is printed at the end, e.g. for national test region `hex12`:

```text
Evaluated 1 hexel(s) against BurnP3+ (area: actual). Mean over hexels:
  target           ccc    spearman         mae        bias
  bp              0.85        0.90       0.001     -0.0005
  fi              0.79        0.86        1210      -187.9
  ros             0.71        0.78       1.323     -0.3188
```

| File | Content |
|---|---|
| `metrics_summary.csv` | Mean of each metric over regions, per target |
| `metrics_per_hexel.csv` | One row per region × target, with the number of cells scored |
| `metrics_per_firezone.csv` | With `--by_firezone`: one row per region × fire zone × target |
| `hazard_metrics_summary.csv`, `hazard_metrics_per_hexel.csv` | Agreement of the hazard classes |
| `hazard_confusion_matrix.{csv,png}` | Hazard classes, BurnP3+ (rows) vs AI surrogate (columns) |
| `plots/hexNN/` | AI surrogate, BurnP3+ and difference maps; scatter plots and histograms |
| `predictions/` | The predictions, when evaluate ran the model |
| `evaluation_manifest.json`, `evaluate.log` | Model, options, input check, warnings and summary |

Extra options: `--by_firezone`, `--metrics ccc mae` (a subset), `--no_plots` (faster),
`--mask_scope buffer` (also scores the region and its buffer ring separately), plus the `predict` options.
When reusing predictions, evaluate warns if they were made with another model.

### Metrics

Metrics are computed per region, over the cells where BurnP3+ has data, then averaged over regions.

| Metric | Meaning (AI surrogate vs BurnP3+) | Best |
|---|---|---|
| `ccc` | Concordance correlation: agreement in value, not just ranking | 1 |
| `spearman` | Rank correlation: are the same places ranked high and low | 1 |
| `mae`, `mse` | Mean absolute / squared error, in the target's units | 0 |
| `normalized_mae` | MAE divided by the mean BurnP3+ value | 0 |
| `bias`, `normalized_bias` | Mean (AI surrogate − BurnP3+); negative = under-prediction | 0 |
| `mae_topXX` | MAE over the top XX% cells of either map (`01` = top 1%) | 0 |
| `iou_topXX` | Overlap of the top XX% cells of both maps (`005` = top 0.5%) | 1 |
| `auc_iou_top10`, `auc_iou_full` | Mean top-K overlap over K = 1–10% and 1–99% | 1 |

Rows with target `hazard` score the raw hazard map. The hazard **class** files report
`exact_accuracy`, `within_1_accuracy` / `within_2_accuracy` (off by at most 1 or 2 classes),
`mean_absolute_class_error`, `macro_iou`, `macro_f1` and per-class IoU/F1.

## 🌲 Regional studies

For a regional study (e.g. a study area prepared as a single `hexNN` folder):

1. **No mask?** Pass `--mask_scope none` to all commands: the whole raster extent is used and
   `mask_grids/` is not needed.
2. **Scenarios?** Pass `--scenario_name NAME` to use `hexNN_fbp_NAME.tif` and `results/NAME/`.
3. **New fuel codes?** Add the fuel tables (see [Fuel codes](#fuel-codes)).
4. **Regional fire sizes?** Pass `--fire_size_table`.

```bash
python -m inference.evaluate --bundle nrcan-surrogate-bp-fi-ros-v1.0 --project path/to/NWT_data \
    --scenario_name FireExcludeSpotting --mask_scope none --output path/to/evaluation
```

> [!IMPORTANT]
> The AI surrogate learned from national BurnP3+ runs. In a region whose fire regime differs, it usually
> gets the **spatial pattern** of burn probability right (high Spearman) but not always its **absolute
> level** (low CCC, large bias); FI and ROS hold up better. Run `evaluate` on a BurnP3+ run of your region
> before relying on absolute burn probabilities.

## 🐍 Using it from Python

```python
from inference.bundle import load_bundle
from inference.check import check_project
from inference.evaluate import run_evaluate
from inference.predict import run_predict

report = check_project(load_bundle("nrcan-surrogate-bp-fi-ros-v1.0"), "path/to/project")
print(report.format())

run = run_predict("nrcan-surrogate-bp-fi-ros-v1.0", "path/to/project", "path/to/predictions", device="cpu")
print(run.hexels[0].outputs["bp"])  # path to the BP GeoTIFF of the first region

evaluation = run_evaluate("nrcan-surrogate-bp-fi-ros-v1.0", "path/to/project", "path/to/evaluation", plots=False)
print(evaluation.summary)           # pandas DataFrame
```

The keyword arguments mirror the command-line options (`hex_ids`, `scenario_name`, `mask_scope`,
`fire_size_table`, `device`, ...).

## ⚙️ Performance

Measured on a single CPU core, no GPU:

| Area | `check` | `predict` | `evaluate` (predict + scoring) |
|---|---|---|---|
| National region `hex12` (~18 M cells) | seconds | ~15 min | + ~7 min scoring with plots, ~6 GB peak memory |
| NWT regional study (~5.7 M cells) | 17 s, 0.9 GB | ~6 min | 12 min total, 3 GB peak memory |

A GPU (CUDA or Apple MPS) speeds up `predict`; `--no_plots` speeds up `evaluate`.

## 🩺 Troubleshooting

| Problem | Fix |
|---|---|
| `check` reports errors | Fix them and re-run `check`; messages name the file concerned and what is wrong |
| Missing mask shapefile | Add `spatial/mask_grids/hexNN_actual.shp`, or use `--mask_scope none` |
| Unknown fuel codes | Add `FuelTypes.csv` and `FuelCodeCrosswalk.csv` to `tabular/`, or recode those cells |
| Out of memory | Lower `--batch_size` (e.g. `2`) |
| Hangs or crashes when loading data on Windows/macOS | Keep `--num_workers 0` (the default) |
| Slow start-up | `--skip_checksums` skips verifying the bundle files |
| Output folder not empty | Choose another `--output` or add `--overwrite` |

## 🔧 For maintainers: exporting a bundle

Turn a training checkpoint into a bundle (run once per released model):

```bash
python -m inference.export_bundle --checkpoint path/to/best.pth --out_dir nrcan-surrogate-bp-fi-ros-v1.0 \
    --name nrcan-surrogate-bp-fi-ros --version 1.0.0 \
    --hazard_denominator_json path/to/hazard_scale_denominator.json \
    --fire_size_table path/to/df_fire_fru_25ha_1970_2023.csv \
    --fire_size_table_note "Source of the fire-size table" \
    --selection_note "How this checkpoint was selected"
```

See `--help` for all options.
