# Counterfactual Fuel Intervention

Evaluate how model predictions (BP, FI, ROS) respond to hypothetical edits of the fuel
map — e.g. "what if this non-fuel barrier were burnable?" or "what if fuel X were
replaced by non-fuel?" — for a fixed set of hexels, using already-trained checkpoints.

## Pipeline

```
configs/counterfactual_fuel.yaml
        │
        ▼
src/evaluate_counterfactual.py            # 1. Evaluate baseline + selected scenario(s), per endpoint
        │
        ▼
src/datasets/postprocessing/counterfactual/plotting/
  counterfactual_fuel_intervention_map.py # 2. Plot the fuel edit itself (original vs. replacement fuel)
  counterfactual_response_maps.py         # 3. Plot GT/baseline/scenario/Δ prediction maps + hotspot patch zoom
  counterfactual_local_zoom_panels.py     #    Fuel-specific zoom on selected evaluated-edit neighborhoods
  counterfactual_change_distribution.py   # 4. Summarize prediction-change attribution and its decay with distance
```

1. **Evaluate** (`evaluate_counterfactual.py`): runs the `baseline` scenario (unmodified
   fuel) and each selected `fuel` scenario, for each selected endpoint. Fuel scenarios
   apply a `FuelCounterfactualTransform` (see `fuel_counterfactual_transform.py` in this folder) that
   edits the fuel channel of each patch before it reaches the model. Writes predicted
   hexel rasters under `<save_dir>/predictions/<scenario>/<endpoint>/`, an index mapping
   `(scenario, endpoint) -> prediction_dir` (`scenario_prediction_index.csv`), evaluation
   metrics (`counterfactual_metrics.csv`), and fuel-edit summaries (`fuel_edit_summary.csv`,
   `fuel_component_replacements.csv`). Fuel scenarios also write the exact baseline and
   edited fuel rasters used by inference under each prediction directory's
   `fuel_intervention/` subdirectory.
2. **Fuel edit map** (`counterfactual_fuel_intervention_map.py`): for one hexel/scenario/
   endpoint, renders the persisted evaluated fuel intervention (original grouped fuel vs.
   replacement fuel groups), plus a per-hexel pixel-count summary CSV.
3. **Response maps** (`counterfactual_response_maps.py`): generic, endpoint-parameterized
   (`--endpoint {bp,fi,ros}`) ground-truth/baseline/scenario/Δ maps for one hexel/scenario
   pair, plus zoom-ins on the highest-Δ patches and per-pixel Δ histogram/concentration
   plots. For `fuel` scenarios it uses the persisted baseline/scenario fuel support:
   non-burnable pixels contribute zero, so newly burnable pixels contribute the scenario
   response and newly non-burnable pixels contribute the negative baseline response.
   Reusable as-is by any future counterfactual scenario family — not just fuel edits.
4. **Local zoom panels** (`counterfactual_local_zoom_panels.py`): fuel-intervention-specific
   companion to the response maps. Instead of generic high-|Δ| hotspots, it selects
   fixed-size windows that contain the persisted evaluated fuel edits and a strong
   direction-aligned hazard/FI response. It renders the edit mask, edited fuel groups, and
   baseline/scenario/Δ FI and hazard (BP × FI) maps for each selected window, plus a
   per-window summary CSV. This supports both non-fuel-to-burnable and
   burnable-to-non-fuel scenarios without reconstructing the intervention.
5. **Change distribution** (`counterfactual_change_distribution.py`): computes per-hexel Δ
   magnitude/sign/concentration statistics — including edited vs. off-edit attribution
   and top-fraction abs-change shares — into a summary CSV, and plots/tabulates mean Δ as a
   function of distance from the edited fuel pixels.

## Configuration (`configs/counterfactual_fuel.yaml`)

```yaml
raw_data_dir: "/path/to/raw/hexel/data"
save_dir: "experiments/counterfactual_fuel_hex16"
hex_ids: ["16"]

endpoints:
  bp:
    config_path: "configs/bp_common_input_pipeline.yaml"   # trained checkpoint's config
  fi:
    config_path: "configs/fi_common_input_pipeline.yaml"
  ros:
    config_path: "configs/ros_common_input_pipeline.yaml"

scenarios:
  - name: "baseline"
    kind: "baseline"
    description: "Unmodified prepared fuel inputs."

  - name: "remove_barriers_adjacent_modal"
    kind: "fuel"
    description: "Replace each connected non-fuel component with its modal adjacent burnable fuel group."
    params:
      mode: "nonfuel_to_burnable_local_adjacent_modal"
      nonfuel_ids: [100, 101, 102, 105, 106, 110]
```

- `endpoints`: one entry per trained model to evaluate; `config_path` points at that
  model's own training/evaluation config (used to resolve its checkpoint, data root, and
  test split).
  - **Single multi-output model**: if one checkpoint jointly predicts `bp`/`fi`/`ros`
    instead of training three separate single-target models, list `bp`, `fi`, and `ros`
    endpoints all pointing at that **same** `config_path` (see
    `configs/counterfactual_fuel_multi_output.yaml`). `evaluate_counterfactual.py`
    detects that the endpoints resolve to the same config + data root and runs inference
    only once per scenario, reusing the resulting prediction directory for every alias
    endpoint. Predicted rasters for a multi-output model are written with a
    target-name suffix (`hexel_XX_bp_predicted.tif`, `hexel_XX_fi_predicted.tif`, ...)
    into that shared directory, and the plotting scripts below resolve the correct
    raster automatically from the `--endpoint` name — no other config or CLI changes are
    needed.
- `scenarios`: exactly one scenario named `baseline` with `kind: "baseline"`, plus any
  number of `fuel` scenarios. Each
  `fuel` scenario's `params` are passed to `apply_fuel_edit` (`counterfactual_fuel.py`),
  keyed by `mode`:

  | `mode` | Direction | Required params | Behaviour |
  |---|---|---|---|
  | `nonfuel_to_burnable_local_adjacent_modal` | barrier removal | `nonfuel_ids` | Each connected non-fuel component is replaced by the modal burnable fuel group among its adjacent pixels. |
  | `nonfuel_to_burnable_adjacent_modal` | barrier removal | `nonfuel_ids` | Same as above, computed globally instead of per connected component. |
  | `nonfuel_to_burnable_fixed` | barrier removal | `nonfuel_ids`, `replacement_fuel_id` | Every non-fuel pixel is replaced by a single fixed fuel id. |
  | `burnable_to_nonfuel` | barrier insertion | `nonfuel_ids`, `insertion_mask` | Replaces burnable pixels under a caller-supplied mask with non-fuel. |
  | `burnable_components_to_nonfuel_random` | barrier insertion | `nonfuel_ids`, `replacement_nonfuel_id`, `target_burnable_area_fraction`, `seed` | Randomly samples whole burnable connected components (weighted by area) until their combined area reaches `target_burnable_area_fraction` of total burnable area, then replaces them with a fixed non-fuel id. |
  | `burnable_to_burnable_fixed` | fuel-type substitution | `nonfuel_ids`, `source_fuel_ids`, `replacement_fuel_id` | Every burnable pixel whose fuel id is in `source_fuel_ids` is replaced by a single fixed burnable fuel id. |

  Adding a new scenario `mode` means adding a branch in `apply_fuel_edit` and a matching
  entry in this table.

## Running

By default, `evaluate_counterfactual.py` runs every endpoint and scenario in the config.
Pass `--endpoint`/`--scenario` (repeatable) to restrict to a subset:

```bash
# Evaluate everything in the config
python -m src.evaluate_counterfactual --config configs/counterfactual_fuel.yaml --overwrite

# Evaluate only the bp endpoint for one scenario (baseline is always included)
python -m src.evaluate_counterfactual --config configs/counterfactual_fuel.yaml \
    --endpoint bp --scenario remove_barriers_adjacent_modal

# Multiple endpoints/scenarios: repeat the flag
python -m src.evaluate_counterfactual --config configs/counterfactual_fuel.yaml \
    --endpoint bp --endpoint fi --scenario remove_barriers_adjacent_modal --scenario remove_barriers_fixed_c2
```

Plotting scripts operate on one hexel/scenario/endpoint at a time and read paths from the
counterfactual config plus the `scenario_prediction_index.csv` written by evaluation.
`--experiment_dir` and `--raw_data_dir` are optional overrides:

```bash
# Plot the fuel intervention map
python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_fuel_intervention_map \
    --config configs/counterfactual_fuel.yaml \
    --scenario remove_barriers_adjacent_modal --endpoint bp --hex_id 16

# Plot GT/baseline/scenario/Δ response maps + patch zoom for one endpoint
python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_response_maps \
    --config configs/counterfactual_fuel.yaml \
    --scenario remove_barriers_adjacent_modal --endpoint fi --hex_id 16

# Plot local zoom panels on selected evaluated-edit neighborhoods (uses bp + fi)
python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_local_zoom_panels \
    --config configs/counterfactual_fuel.yaml \
    --scenario remove_barriers_adjacent_modal --hex_id 16

# Summarize the prediction change distribution
python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_change_distribution \
    --config configs/counterfactual_fuel.yaml \
    --scenario remove_barriers_adjacent_modal --endpoint bp
```

Run `--help` on any script for the full set of options (e.g. `--zone_overlay` to draw
firezone boundaries, `--downsample` for lower-resolution map rendering).

Submit `run_files/counterfactual_fuel_iROS.sh` for GPU evaluation, then
`run_files/counterfactual_fuel_plots.sh` for all configured intervention, response,
local-zoom, and change-distribution plots.
