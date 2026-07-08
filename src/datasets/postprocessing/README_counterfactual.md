# Fixed-model input counterfactuals

We hold a trained model fixed, edit one slice of its **inputs** (fuel, FWI, or
wind regime) for one focus hex, re-run inference, and measure the paired change
in the model's output. This directory implements that pipeline.

## The flow

```
configs/counterfactual_hex16.yaml      declare scope: hex, endpoints, scenarios
        │
        ▼
counterfactual_materialize.py          build one isolated data root per
                                       (scenario × endpoint), edit the inputs,
                                       link the checkpoint, write a generated
                                       config + scenario_prediction_index.csv
        │
        ▼
python -m src.evaluate_hexels          run inference per generated config
  (run_files/eval_hexels.sh, SLURM)    → <prediction_dir>/predicted_hexels/*.tif
        │
        ▼
counterfactual_compare.py              paired Δ(scenario − baseline) metrics
        │
        ▼
counterfactual_*_map.py                per-scenario figures
```

## Modules

Edit logic (one per scenario `kind`):
- `counterfactual_fuel.py` — fuel-grid edits (`kind: fuel`)
- `counterfactual_fwi.py` — daily FWI-regime swap (`kind: fwi`)
- `counterfactual_wind_regime.py` — per-zone wind regime transplant (`kind: wind_regime`)
- `counterfactual_wind_direction.py` — uniform wind-direction rotation (`kind: wind_direction`), either hex-wide or per-zone
- `counterfactual_weather.py` — wind re-encoding + roundtrip validation shared by the weather kinds

A `composite` kind applies one fuel edit **and** one weather edit in a single
scenario by listing sub-edits; the materializer dispatches each to its existing
edit path (no new edit logic). See "Adding a new scenario".

Orchestration & spec:
- `counterfactual.py` — config dataclasses, loader, wind geometry
- `counterfactual_materialize.py` — dispatches `scenario.kind` → edit fn, builds data roots
- `counterfactual_compare.py` — paired raster-delta summaries

Figures (each consumes the predictions for its scenario family):
- fuel: `counterfactual_hazard_map.py`, `counterfactual_fi_map.py`,
  `counterfactual_fuel_intervention_map.py`, `counterfactual_local_zoom_panels.py`
- fwi: `counterfactual_fwi_map.py`
- wind_regime: `counterfactual_wind_regime_map.py`
- wind_direction: `counterfactual_wind_direction_map.py` (paired dominant vs +180° mirror; add
  `--direction_scope zone` for per-zone dominant/opposite scenarios)
- any BP scenario: `counterfactual_bp_scenario_map.py` (generic single-scenario
  response maps + patch zoom + delta histogram)
- any FI scenario: `counterfactual_fi_scenario_map.py` (generic single-scenario
  response maps + patch zoom + delta histogram)
- any ROS scenario: `counterfactual_ros_scenario_map.py` (generic single-scenario
  response maps + patch zoom + delta histogram; use for `composite` scenarios). For
  fuel-editing scenarios it masks non-fuel in the ground-truth/baseline panels but
  keeps the filled pixels in the scenario panel, with baseline treated as zero ROS
  there so Δ shows the full barrier-removal effect.

Shared helpers (keep dependency-light, no scenario-specific logic):
- `counterfactual_viz.py` — IO/plot hub, incl. `plot_delta_histogram`
  (log-count histogram + |Δ| concentration curve) reusable across interventions
  and `overlay_zone_boundaries` (firezone borders as a `LineCollection`, drawn on
  every hex map; clip to the displayed hexagon via `load_zone_labels(..., support=)`)
- `counterfactual_ros_maps.py` — ROS response-map + GT patch-zoom plotting shared
  by the wind_regime and wind_direction figure scripts
- `counterfactual_weather_maps.py` — spatialized-weather map IO, incl.
  `load_zone_labels` (firezone raster on the prediction grid)
- `fuel_barrier_geometry.py` — fuel grouping + barrier-relative geometry

Every `*_map.py` writes its PNGs under `experiments/<exp>/figures/<group>/`
(`fwi_daily`, `wind_direction`, `wind_zone_peak`, `<scenario>_bp`/`<scenario>_fi`,
`<scenario>` for generic ROS,
`fuel_fi`, `fuel_intervention`, `hazard`, `fuel_local_zoom`); override with
`--out_dir`. Summary CSVs stay at the experiment-dir root.

## Running it

The unit of work is one `(scenario × endpoint)` pair, so scope to exactly what
you need. Below runs a single scenario end-to-end; drop the `--scenario`/
`--endpoint` flags to sweep all of that axis.

```bash
# 1. Materialize inputs for one scenario (optionally one endpoint)
python -m src.datasets.postprocessing.counterfactual_materialize \
    --scenario fwi_daily_low_to_high --endpoint fi --overwrite

# 2. Inference (SLURM) — one eval per generated config this scenario produced
sbatch run_files/eval_hexels.sh   # point it at the generated config(s)

# 3. Paired metrics (reads whatever is in scenario_prediction_index.csv)
python -m src.datasets.postprocessing.counterfactual_compare

# 4. Figures for that scenario's family (see the module map above)
python -m src.datasets.postprocessing.counterfactual_fwi_map
```

Firezone boundaries are opt-in on counterfactual map scripts:

```bash
python -m src.datasets.postprocessing.counterfactual_wind_direction_map \
    --zone_overlay \
    --zone_overlay_linewidth 1.4 \
    --zone_overlay_color "#111111"
```

Season-conditioned high-FWI donor-row scenarios use the existing external
transplant mode with `season_values`. For the current leaf-off probe,
`bc_spring_high_fwi_transplant` selects the highest-FWI BC donor row with
`Season == s1`; `bc_nonspring_high_fwi_transplant` selects from `s2/s3`. Render
BP/FI after inference with:

```bash
python -m src.datasets.postprocessing.counterfactual_bp_scenario_map \
    --scenario bc_spring_high_fwi_transplant --label "BC spring high-FWI transplant" --zone_overlay
python -m src.datasets.postprocessing.counterfactual_fi_scenario_map \
    --scenario bc_spring_high_fwi_transplant --label "BC spring high-FWI transplant" --zone_overlay
```

For the cleaner within-hex seasonal diagnostic, use the zone-conditioned
scenarios `hex16_spring_zone_high_fwi_transplant` and
`hex16_nonspring_zone_high_fwi_transplant`. These select the highest-FWI row
within each hex16 `WeatherZone` and season set, then transplant that row back
only into the same zone.

Scoping notes:
- `--scenario` and `--endpoint` are repeatable (`--scenario a --scenario b`);
  omitting a flag = all scenarios / all enabled endpoints.
- The full sweep is just the no-flag default:
  `python -m ...counterfactual_materialize --overwrite`.
- Targeted re-materialization is safe: `scenario_prediction_index.csv` is
  **merged**, not overwritten, so untouched scenarios keep their index rows.
- `--overwrite` lets a re-run replace existing generated files; without it the
  step errors if any target already exists (a guard against accidental reruns).

## Adding a new scenario

1. **Pick or add a `kind`.** Reusing an existing kind = config-only:
   ```yaml
   - name: my_scenario
     kind: fwi
     description: ...
     params: { mode: daily_regime_swap, direction: low_to_high, ... }
   ```
2. **New kind** = add an `apply_<kind>_scenario(raw, processed, stats, params, *, seed)`
   module (return `(edited_processed_hex, edit_report)`) and one dispatch branch in
   `counterfactual_materialize._write_weather_table` (or the fuel branch in
   `_materialize_one` for grid edits).
3. **Composite** = combine one existing fuel edit with one existing weather edit
   (no new edit logic). List the sub-edits; each is dispatched to its own path:
   ```yaml
   - name: remove_barriers_wind_opposite
     kind: composite
     params:
       edits:
         - { kind: fuel, params: { mode: nonfuel_to_burnable_local_adjacent_modal } }
         - { kind: wind_direction, params: { mode: uniform_direction, offset_deg: 180.0 } }
   ```
4. **Figures** — reuse a `*_map.py` if the response view fits, else add one that
   reads `scenario_prediction_index.csv` + the shared helpers above.

> Inference runs the live tree from `$SLURM_SUBMIT_DIR`. Do not edit `src/` while
> a counterfactual eval job is pending/importing.
