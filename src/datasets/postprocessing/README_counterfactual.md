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
- `counterfactual_weather.py` — wind re-encoding + roundtrip validation shared by the weather kinds

Orchestration & spec:
- `counterfactual.py` — config dataclasses, loader, wind geometry
- `counterfactual_materialize.py` — dispatches `scenario.kind` → edit fn, builds data roots
- `counterfactual_compare.py` — paired raster-delta summaries

Figures (each consumes the predictions for its scenario family):
- fuel: `counterfactual_hazard_map.py`, `counterfactual_fi_map.py`,
  `counterfactual_fuel_intervention_map.py`, `counterfactual_local_zoom_panels.py`
- fwi: `counterfactual_fwi_map.py`
- wind_regime: `counterfactual_wind_regime_map.py`

Shared helpers (keep dependency-light, no scenario-specific logic):
- `counterfactual_viz.py` — IO/plot hub
- `counterfactual_weather_maps.py` — spatialized-weather map IO
- `fuel_barrier_geometry.py` — fuel grouping + barrier-relative geometry

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
3. **Figures** — reuse a `*_map.py` if the response view fits, else add one that
   reads `scenario_prediction_index.csv` + the shared helpers above.

> Inference runs the live tree from `$SLURM_SUBMIT_DIR`. Do not edit `src/` while
> a counterfactual eval job is pending/importing.
