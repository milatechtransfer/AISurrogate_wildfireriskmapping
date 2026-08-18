# Scenario-Driven Spread Opportunity

## Purpose

NRCan calibrated BurnP3+ with spread-event days and daily burning hours to
reproduce historical fire-size distributions. The scenario-driven models use
those simulator controls instead of supplying final fire size directly:

```text
total burning hours = spread-event days * daily burning hours
```

The implementation is on `exp/scenario-spread-opportunity-v3`. It preserves
the existing fire-size mechanistic v2 and adds:

- a conventional `512 -> 256` U-Net using min-max-normalized q10/q50/q90
  total-burning-hour channels;
- mechanistic v2.1, which keeps fire-size q3 stopping but adds stability
  controls;
- mechanistic v3, which propagates separate q10/q50/q90 time-budget cohorts.

## Data preparation

The complete processor is:

```text
data_preparation/process_spread_opportunity.py
```

For each hex it reads:

```text
hexXX_SpreadEventDays.csv
hexXX_DailyBurningHours.csv
hexXX_ScenarioDistributions - <letter> - FINAL.csv
hexXX_IgnitionDistribution.csv
hexXX_FireZones.csv
spatial/firezones_grid.tif
```

Path resolution lives in `data_preparation/paths.py`. The scenario filename
must be discovered dynamically because its letter differs across hexes.

### PMF construction

For hex \(h\), fire response unit \(z\), season \(s\), spread days \(D\), and
daily hours \(H\), the seasonal total-hour PMF is the product distribution:

\[
T_{h,z,s}=D_{h,z,s}H_{h,s}.
\]

The implementation uses the full cross product, so it remains correct if daily
burning hours are represented by a distribution rather than a single mean.
Duplicate products are combined and all relative frequencies are normalized.

Ignition likelihood is first summed across causes:

\[
L_{h,z,s}=\sum_c L_{h,z,s,c}.
\]

Valid seasons are then normalized within each `(hex, FRU)` and their complete
PMFs are mixed. Quantiles are extracted only after mixing; weighted averages of
seasonal quantiles are not equivalent.

### Fallback

Some raster zones do not have a direct positive-likelihood scenario mixture.
The processor constructs one explicit fallback PMF per hex using:

```text
FRU raster area fraction * ignition likelihood
```

Missing zones receive its statistics and `SCENARIO_FALLBACK=1`. Direct zones
receive `SCENARIO_FALLBACK=0`. Missing scenario components with positive
likelihood are reported in `SCENARIO_DROPPED_LIKELIHOOD` rather than silently
treated as valid.

Known source-data anomalies handled explicitly:

- the scenario filename letter varies by hex;
- hex28 has a malformed first scenario-table header;
- relative frequencies need not sum to 100;
- blank scenario seasons are ignored with warnings;
- `hex30/fru13/s2` has positive ignition likelihood without a valid spread
  mapping and is reported as dropped likelihood;
- three raster IDs absent from local `FireZones.csv` are emitted as
  fallback-only rows: hex46/1, hex49/61, and hex51/5.

### Output and normalization

The generated table is:

```text
spread_opportunity_processed.csv
```

Its important columns are:

```text
hex_id
GRIDCODE
FIREZONE
TOTAL_BURN_HOURS_MEAN
TOTAL_BURN_HOURS_Q10
TOTAL_BURN_HOURS_Q50
TOTAL_BURN_HOURS_Q90
NORM_TOTAL_BURN_HOURS_MEAN
NORM_TOTAL_BURN_HOURS_Q10
NORM_TOTAL_BURN_HOURS_Q50
NORM_TOTAL_BURN_HOURS_Q90
SCENARIO_FALLBACK
SCENARIO_SEASON_COUNT
SCENARIO_DROPPED_LIKELIHOOD
```

One shared linear min-max transform is fitted from q10/q50/q90 values belonging
to training hexes. That same range is used for the mean and is applied to
validation and test hexes without clipping. The fitted range is persisted in:

```text
spread_opportunity_norm_params.json
```

This is leakage-safe and preserves the physical spacing among quantiles. The
current data range fitted from the training split is `[1, 140]` hours.

Run the processor with:

```bash
bash run_files/context_models/prepare_spread_opportunity.sh
```

The hex01/fru21 regression fixture is:

```text
mean = 9.736
q10  = 4
q50  = 6
q90  = 18
```

## Dataset integration

`spatialized_spread_opportunity` is registered in:

```text
src/datasets/utils.py
src/config.py
```

It reuses `SpatializedTabularSource`, keyed by both `hex_id` and `GRIDCODE`.
Configurations use `missing_value_strategy: raise` so incomplete patch lookup
coverage fails instead of receiving an unrelated global mean. The q3 values
are already precomputed; do not configure the source's empirical `quantiles`
option.

The conventional U-Net configuration is:

```text
configs/context_models/unet_512_crop_256_spread_opportunity_q3.yaml
```

It adds exactly three predictive channels:

```text
NORM_TOTAL_BURN_HOURS_Q10
NORM_TOTAL_BURN_HOURS_Q50
NORM_TOTAL_BURN_HOURS_Q90
```

The fallback mask remains in the CSV for auditability but is intentionally
excluded from this pure min-max U-Net ablation.

## Mechanistic v2.1 stability controls

Configuration:

```text
configs/mechanistic/mechanistic_propagation_v21_512_crop_256_firesize_q3.yaml
```

Mechanistic v2.1 retains the v2 fire-size q3 stopping rule. It addresses the
observed physical-BP calibration drift by:

- detaching reach/transmission features before the FI/ROS behavior head;
- using base learning rate `3e-4`;
- applying `0.1x` LR to mechanistic scalar parameters;
- applying `0.25x` LR to local BP calibration;
- restricting local calibration to approximately `0.8x-1.25x`;
- clipping gradient norm at `1.0`;
- selecting checkpoints by the mean BP/FI/ROS CCC.

The trainer support for parameter groups, clipping, derived `mean/ccc`, and
diagnostic scalar logging is in `src/trainer.py`.

## Mechanistic v3 time budgets

Implementation:

```text
src/models/mechanistic_propagation.py
    DifferentiableTimeBudgetPropagation
    MechanisticFirePropagationUNet
```

Configuration:

```text
configs/mechanistic/mechanistic_propagation_v3_512_crop_256_spread_opportunity_q3.yaml
```

The normalized q3 channels are converted back to physical hours with the saved
training range. V3 then runs one propagation cohort per quantile. At step \(k\)
with hours per step \(\Delta t\), cohort \(q\) uses:

\[
g_{q,k}=\sigma\left(\frac{T_q-k\Delta t}{\tau}\right).
\]

The current settings are:

```text
24 propagation steps
2 hours per step
2-hour gate temperature
q10/q50/q90 cohort weights = 0.3/0.4/0.3
```

Each quantile cohort retains its own reach state through the propagation loop,
and the final reach probability is their weighted sum.

This remains a practical approximation: the time-budget channels are
FRU-rasterized, so the active budget can change when a frontier crosses into a
different FRU. The current network does not preserve individual ignition
identity and therefore cannot yet attach one sampled scenario budget to each
fire for its full lifetime. A future implementation would need separate
ignition cohorts in addition to the current quantile cohorts.

The fallback mask is supplied as an encoder context channel but is not parsed
as a time-budget quantile.

## Experiments

Full conventional U-Net:

```bash
bash run_files/context_models/submit_spread_opportunity_unet.sh
```

Five-epoch mechanistic pilots:

```bash
bash run_files/context_models/submit_mechanistic_scenario_pilots.sh
```

Pilot configurations:

```text
configs/mechanistic/mechanistic_propagation_v21_512_crop_256_firesize_q3_pilot.yaml
configs/mechanistic/mechanistic_propagation_v3_512_crop_256_spread_opportunity_q3_pilot.yaml
```

The v2.1 pilot isolates stabilization from the scenario-budget change. The v3
pilot tests the combined stabilized model and scenario-hours propagation.

## Tests

Relevant coverage is in:

```text
tests/test_spread_opportunity.py
tests/test_dataloader.py
tests/test_mechanistic_propagation.py
tests/test_configs.py
tests/test_trainer.py
```

These tests cover PMF arithmetic, the fixed hex01 fixture, malformed headers,
fallback construction, shared train-only min-max fitting, source
registration, time-budget monotonicity, quantile cohort weighting, architecture
construction, optimizer LR groups, and `mean/ccc`.
