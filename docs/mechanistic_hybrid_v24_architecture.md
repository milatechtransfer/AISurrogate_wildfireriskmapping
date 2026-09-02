# Mechanistic hybrid v2.4: engineering architecture and pipeline

This document describes the released v2.4 model as implemented. It is an
architecture and data-flow reference for engineers; operational commands and
the regional-data contract remain in
[`mechanistic_hybrid_v24.md`](mechanistic_hybrid_v24.md).

## Released reference

| Item | Value |
| --- | --- |
| Release branch | `release/mechanistic-hybrid-v24` |
| Release commit | `04c2c83` |
| Training implementation commit | `e7570ca` |
| Reference config | `configs/mechanistic_hybrid_v24_reference.yaml` |
| Model class | `src/models/mechanistic_hybrid_v24.py::MechanisticHybridV24` |
| Prepared data | `/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v6_native_context_512_crop_256_ignition_probability_mass_log_firesize` |
| Checkpoint | `/network/projects/amlrt/nrcan_wildfires/checkpoints/burnp3plus/final_experiments/mechanistic_hybrid_v24_native_512_crop_256_firesize_q3/best.pth` |
| Selected epoch | 27 |
| Selection metric | Stitched validation `hex/mean/ccc = 0.772381` |
| Trainable parameters | 8,440,543 |
| State-dict tensors | 104 |

The checkpoint strictly loads into the release implementation. It predicts BP,
FI, and ROS jointly, but BP and behavior use separate decoders after the
shared encoder.

## End-to-end map

```text
Raw BurnP3+ rasters and tables
  |
  | data_preparation/process_hexels_into_grids.py
  v
Aligned native-grid 512x512 patches + metadata CSVs
  |
  | src/datasets/dataset.py::MultiSourceDataset
  | src/datasets/sources/grids.py::GridSource
  | src/datasets/sources/spatialized_tabular.py::SpatializedTabularSource
  v
12-channel spatial tensor + 36-channel iROS/HFI fuel tensor
  |
  | src/models/factory.py::build_model
  v
MechanisticHybridV24
  |
  +--> shared CNN encoder
  |
  +--> 1/8-resolution physical travel-time model
  |      |
  |      +--> physical per-fire reach
  |      +--> ignition-count aggregation
  |      +--> coarse physical BP
  |
  +--> BP decoder --> asymmetric signed hazard repair --> BP logits
  |
  +--> behavior decoder --> physical FI/ROS references --> FI and ROS
  |
  | src/trainer.py::Trainer
  v
Centered 256x256 supervision + task and physical regularization losses
  |
  v
Atomic best.pth / last.pth checkpoints
  |
  +--> src/evaluate_hexels.py         stitched national evaluation
  +--> inference/predictor.py         tensor-level checkpoint inference
  +--> inference/run_ai_surrogate_model_hexel_inference.py
                                      preparation, stitching, GeoTIFFs
```

## File-level pipeline

| Stage | Primary files | Responsibility |
| --- | --- | --- |
| Configuration | `configs/mechanistic_hybrid_v24_reference.yaml`, `src/config.py`, `src/config_io.py` | Defines geometry, channels, physical constants, losses, optimization, and release paths. |
| Patch preparation | `data_preparation/process_hexels_into_grids.py` | Creates native-grid context windows and records input/target geometry in `meta_hex_*.csv`. |
| Weather preparation | `data_preparation/tabular/weather.py` | Builds processed weather features, including Cartesian wind components. |
| Fire-size and count preparation | `data_preparation/utils.py`, inference preparation helpers | Builds direct log-hectare fire-size quantiles, missing-zone fallback artifacts, and ignition-count summaries. |
| Dataset assembly | `src/datasets/dataset.py` | Loads a patch once, asks each source for its view, concatenates spatialized sources, and preserves auxiliary tensors. |
| Grid and fuel source | `src/datasets/sources/grids.py` | Extracts ignition/elevation/targets and converts categorical fuel codes into continuous iROS/HFI curves. |
| Zone-level sources | `src/datasets/sources/spatialized_tabular.py` | Rasterizes weather, fire-size, and ignition-count tables by fire-weather zone. |
| Dimension/name discovery | `src/datasets/utils.py` | Resolves the exact spatial channel count, auxiliary dimensions, and semantic channel names. |
| Model construction | `src/models/factory.py` | Dispatches `mechanistic_hybrid_v24` and passes semantic names and fuel-curve statistics. |
| Physical/CNN core | `src/models/mechanistic_hybrid_v22.py` | Implements the shared encoder, fuel interpolation, source cohorts, travel time, fire-size budgets, count model, and behavior references. |
| Direct fire-size contract | `src/models/mechanistic_hybrid_v23.py` | Uses direct `log10(1 + hectares)` channels and an explicit missing-fire-zone mask. |
| V2.4 task split and BP repair | `src/models/mechanistic_hybrid_v24.py` | Separates BP and behavior decoders and introduces the signed BP hazard residual. |
| Training | `src/train.py`, `src/trainer.py`, `src/losses.py` | Builds loaders/model, applies centered supervision, computes losses, selects the best stitched checkpoint, and saves atomically. |
| Evaluation | `src/evaluate_hexels.py`, `src/datasets/postprocessing/` | Reconstructs full hexels, computes patch/stitched metrics, and writes prediction artifacts. |
| Inference | `inference/predictor.py`, `inference/run_ai_surrogate_model_hexel_inference.py` | Loads the checkpoint, prepares or reuses regional patches, predicts, stitches, and saves GeoTIFFs. |
| Release verification | `src/verify_mechanistic_hybrid_v24_release.py`, `tests/test_mechanistic_hybrid_v24*.py` | Checks the bundle, strict state loading, fixed-input equivalence, gradient separation, and release configuration. |

## Spatial geometry

The reference model uses native 100 m cells:

$$
512 \times 100\ {\rm m} = 51.2\ {\rm km}
$$

The supervised output is the centered 256x256 region:

$$
256 \times 100\ {\rm m} = 25.6\ {\rm km}
$$

The input therefore supplies a 128-pixel, 12.8 km halo on every side of the
target. The CNN and physical model produce 512x512 fields, after which
`Trainer._step()` center-crops predictions, targets, and masks to 256x256.

The physical grid is downsampled by eight:

| Quantity | Resolution |
| --- | ---: |
| Input and model output | 512x512 at 100 m |
| Physical propagation grid | 64x64 at 800 m |
| Supervised region on physical grid | 32x32 |
| Propagation relaxation steps | 16 |
| Source tiles | 4x4, yielding 16 source cohorts |

`src/trainer.py::validate_mechanistic_normalization_params` ties the 16
propagation steps to the 16-cell coarse context margin. This is a release
geometry constraint, not a mathematical requirement of the travel-time
solver.

## Input contract

### Spatial tensor

The published checkpoint resolves exactly 12 semantic spatial channels:

| Index | Channel | Meaning |
| ---: | --- | --- |
| 0 | `grid/ignition_grid_human` | Human-caused ignition probability mass. |
| 1 | `grid/ignition_grid_lightning` | Lightning-caused ignition probability mass. |
| 2 | `grid/elevation_grid` | Elevation normalized using frozen training statistics. |
| 3 | `spatialized_weather/InitialSpreadIndex` | ISI normalized using frozen training statistics. |
| 4 | `spatialized_weather/wind_x` | East-west wind component under the training convention. |
| 5 | `spatialized_weather/wind_y` | North-south wind component under the training convention. |
| 6 | `spatialized_fire_size/LOG_SIZE_HA_q10` | Zone q10 of `log10(1 + SIZE_HA)`. |
| 7 | `spatialized_fire_size/LOG_SIZE_HA_q50` | Zone q50 of `log10(1 + SIZE_HA)`. |
| 8 | `spatialized_fire_size/LOG_SIZE_HA_q90` | Zone q90 of `log10(1 + SIZE_HA)`. |
| 9 | `spatialized_fire_size/missing_firezone_mask` | One where the zone used the frozen global fire-size fallback. |
| 10 | `spatialized_ignition_count/NORM_LOG1P_IGNITION_COUNT_MEAN` | Normalized log mean ignition count. |
| 11 | `spatialized_ignition_count/NORM_IGNITION_COUNT_CV` | Normalized ignition-count coefficient of variation. |

Weather preprocessing uses:

$$
w_x = {\rm WindSpeed}\sin({\rm WindDirection}), \qquad
w_y = {\rm WindSpeed}\cos({\rm WindDirection})
$$

The source data use meteorological "from" direction. The model negates the
vector before using it as a travel direction. Regional inference must rotate
the vector into raster-grid coordinates when grid convergence is material.

Zone-level lookup tables are keyed by `(hex_id, zone)` when `hex_id_col` is
configured. This prevents rows from different train/validation/test hexels
from being pooled merely because they share a zone identifier.

The fire-size quantile channels deliberately have two representations inside
the model. V2.3/v2.4 standardize them with the frozen neural mean and standard
deviation before the CNN encoder, while the physical path retains their
direct `log10(1 + hectares)` values.

### Fuel tensor

Raw categorical fuel IDs are not fed into the neural tensor. `GridSource`
looks up two continuous fuel-response curves at each pixel:

- 18 iROS values at ISI bins from approximately 0 to 85;
- 18 HFI values at the same bins.

The auxiliary tensor therefore has 36 channels:

$$
C_{\rm fuel}(p) =
[{\rm iROS}_1,\ldots,{\rm iROS}_{18},
  {\rm HFI}_1,\ldots,{\rm HFI}_{18}]
$$

Separate `FuelCurveEncoder` modules compress the ROS and HFI halves into an
eight-channel learned embedding. The spatial tensor and fuel embedding form
the CNN input.

### Targets

| Target | Model output | Training target space |
| --- | --- | --- |
| BP | Logit | Raw probability |
| FI | Unbounded regression value | Standardized `log1p(FI)` |
| ROS | Unbounded regression value | Standardized `log1p(ROS)` |

## Shared CNN encoder

The encoder is inherited from `MechanisticHybridV22`. With the reference
`propagation_base_channels: 48`, its feature widths are:

```text
512x512: 48 channels
256x256: 96 channels
128x128: 192 channels
 64x64:  288 channels
```

The full-resolution block receives the 12 spatial channels plus the
eight-channel fuel embedding. The 64x64 representation is both the deepest
CNN feature map and the resolution of the physical model.

V2.4 has one shared encoder but two task-specific decoding paths:

- the BP decoder receives differentiable mechanistic fields;
- the behavior decoder receives detached mechanistic fields;
- FI/ROS gradients still update the shared encoder and behavior decoder, but
  do not update the physical speed correction, quantile weights, ignition
  reach scale, or BP decoder.

This separation is the central difference between v2.3 and v2.4.

## Physical propagation model

The following symbols are used below:

- $p$: a 100 m fine pixel;
- $i,j$: 800 m coarse cells;
- $d$: one of eight neighbor directions;
- $s$: one of 16 ignition source cohorts;
- $q$: one of q10, q50, q90 fire-size scenarios.

### 1. Fuel-specific physical ROS and HFI

The model first restores physical ISI:

$$
{\rm ISI}_p =
\mu_{\rm ISI} + \sigma_{\rm ISI}{\rm ISI}^{norm}_p
$$

It linearly interpolates the pixel's iROS and HFI curves at that ISI:

$$
r_p = {\rm interp}(C^{ROS}_p, {\rm ISI}_p), \qquad
f_p = {\rm interp}(C^{HFI}_p, {\rm ISI}_p)
$$

Burnability is defined from the ROS curve:

$$
b_p = {\bf 1}\left[\max_k C^{ROS}_{p,k} > 0\right]
$$

Within coarse cell $i$, base ROS is a harmonic mean over burnable pixels,
multiplied by the burnable fraction:

$$
\bar r_i =
\left(
\frac{\sum_{p\in i} b_p}
{\sum_{p\in i} b_p/\max(r_p,r_{min})}
\right)
\left(
\frac{1}{64}\sum_{p\in i}b_p
\right)
$$

HFI is a burnability-weighted arithmetic mean. This avoids allowing a few
very fast pixels to dominate a fragmented coarse cell.

### 2. Wind anisotropy

Let $\hat e_d$ be the unit vector for direction $d$, and let the
meteorological wind vector be negated to obtain its travel direction. Then:

$$
a_{i,d} = \hat w_i \cdot \hat e_d
$$

$$
s_i = \frac{\lVert w_i\rVert}
{\lVert w_i\rVert + 10}
$$

$$
F^{wind}_{i,d}
=
\exp[-\alpha_w s_i(1-a_{i,d})]
$$

$\alpha_w$ is learned, constrained to `[0, log(8)]`, and initialized at
`log(4)`. Because ISI already contains a wind-speed effect, this term only
attenuates off-axis spread; it does not multiply downwind ROS above the
interpolated iROS value.

### 3. Terrain and learned directional correction

For an incoming edge from $j$ to $i$:

$$
g_{j\rightarrow i}
=
{\rm clip}
\left(
\frac{E_i-E_j}{\ell_{ji}},
-0.5,0.5
\right)
$$

$$
F^{slope}_{j\rightarrow i} = \exp(3g_{j\rightarrow i})
$$

A small CNN receives the deepest encoder features and six physical context
channels:

1. scaled `log1p` coarse base ROS;
2. coarse wind x;
3. coarse wind y;
4. coarse normalized elevation;
5. coarse burnable fraction;
6. relative coarse ignition density.

It predicts one log-speed correction per direction:

$$
\delta^{speed}_{i,d}
=
\log(2)\tanh(c_{i,d})
$$

so the learned multiplier lies in `[0.5, 2]`.

The directional speed at an endpoint is:

$$
v_{i,d}
=
\max\left[
r_i
F^{wind}_{i,d}
F^{slope}_{j\rightarrow i}
\exp(\delta^{speed}_{i,d}),
r_{min}
\right]
$$

### 4. Edge travel time

Cardinal edges are 800 m and diagonal edges are $800\sqrt{2}$ m. The
trapezoidal edge time is:

$$
\Delta t_{j\rightarrow i}
=
\frac{\ell_{ji}}{120}
\left(
\frac{1}{v_{j,d}}+\frac{1}{v_{i,d}}
\right)
$$

The denominator is `2 x 60`: half the edge at each endpoint speed, followed
by conversion from minutes to hours. Off-grid and non-burnable edges receive
a large sentinel travel time.

### 5. Ignition source cohorts

The human and lightning channels are summed and restored from their
probability-mass scale:

$$
L_i =
\sum_{p\in i}
\frac{I^{human}_p+I^{lightning}_p}{10^6}
b_p
$$

The 64x64 physical grid is divided into a 4x4 source grid. In each tile, the
nearest positive cell to the ignition-mass centroid becomes the
representative source. Its weight is the tile's exact ignition mass.

This produces 16 source indices and weights while preserving the spatial
distribution better than a single global source. The weights are scaled down
only as a numerical guard if total context ignition mass exceeds one.

### 6. Differentiable min-travel-time solve

For each source cohort:

$$
T_s^{(0)}(i) =
\begin{cases}
0, & i=s\\
\infty, & i\ne s
\end{cases}
$$

Each of 16 relaxation steps applies:

$$
T_s^{(k+1)}(i)
=
\min
\left[
T_s^{(k)}(i),
\min_{j\in{\cal N}(i)}
\left(T_s^{(k)}(j)+\Delta t_{j\rightarrow i}\right)
\right]
$$

The iteration count is a computational path-length cap. Fire size, not a
fixed number of burn hours, determines the physical reach budget.

### 7. Fire-size budgets

V2.4 directly receives:

$$
z_{s,q} = \log_{10}(1+A_{s,q})
$$

and restores hectares:

$$
A_{s,q} = 10^{z_{s,q}}-1
$$

The equivalent circular radius is:

$$
R_{s,q} =
\sqrt{\frac{10{,}000A_{s,q}}{\pi}}
$$

Using the base ROS at the source, the available travel time is:

$$
B_{s,q}
=
\frac{R_{s,q}}
{60\max(r_s,r_{min})}
$$

The source reference speed is detached before this calculation. Fire-size
budgeting therefore cannot reduce its own loss by changing the source-speed
denominator, while the differentiable edge travel times still train the
learned directional corrections.

Reach is a soft comparison of budget and path time:

$$
Q_{s,q}(i)
=
\sigma\left(
\frac{B_{s,q}-T_s(i)}
{\tau_B}
\right)b_i
$$

where $\tau_B=2$ hours in the reference config. A downward-only area
correction prevents the coarse reach field from implying more burned area
than the source fire-size scenario.

### 8. Source and quantile aggregation

Source cohorts are mixed with their ignition-mass weights:

$$
Q_q(i)=\sum_s w_s Q_{s,q}(i)
$$

The q10/q50/q90 weights are learned on a simplex:

$$
\pi_q = {\rm softmax}(\theta_q)
$$

They are initialized at `[0.3, 0.4, 0.3]`, the probability masses implied by
the q10/q50/q90 intervals, and regularized toward that prior with KL
divergence:

$$
{\cal L}_{quantile}
=
\lambda_q
\sum_q
\pi_q\log\frac{\pi_q}{\pi_q^{prior}}
$$

The per-fire reach is:

$$
q(i)=\sum_q \pi_q Q_q(i)
$$

### 9. Ignition-count distribution and physical BP

The model restores a mean ignition count $\mu_i$ and coefficient of
variation $c_i$. The implied variance is:

$$
\sigma_i^2=(c_i\mu_i)^2
$$

First, a learned global reach scale $s_r$ transforms per-fire reach:

$$
\tilde q_i = 1-(1-q_i)^{s_r}
$$

The no-burn probability uses a moment-matched count distribution:

$$
\log P_0 =
\begin{cases}
-\mu\tilde q,
& \sigma^2\approx\mu
\quad\text{(Poisson)}\\
-k\log(1+\mu\tilde q/k),
\quad k=\mu^2/(\sigma^2-\mu),
& \sigma^2>\mu
\quad\text{(negative binomial)}\\
n\log(1-p\tilde q),
\quad n=\mu^2/(\mu-\sigma^2),\ p=\mu/n,
& \sigma^2<\mu
\quad\text{(binomial)}
\end{cases}
$$

The coarse physical burn probability is:

$$
p^{phys}_{BP}=1-\exp(\log P_0)
$$

## Task-specific decoding

The physical model exposes three 64x64 fields to each decoder:

- per-fire reach $q$;
- coarse physical BP $p^{phys}_{BP}$;
- mean learned log-speed correction.

These are fused with the 288-channel deepest CNN feature map. Three
skip-connected decoder blocks return to full 512x512 resolution.

### BP branch

The BP decoder receives live mechanistic fields. BP loss can therefore update
the physical correction fields, physical scalar parameters, shared encoder,
and BP decoder.

### FI/ROS behavior branch

The behavior decoder is a deep copy of the BP decoder. Its mechanistic inputs
are detached:

```text
per-fire reach.detach()
coarse BP.detach()
mean speed correction.detach()
```

The physical full-resolution HFI, ROS, and speed-correction reference fields
are also detached before the final behavior head. FI/ROS losses therefore
train:

- the shared CNN encoder;
- the behavior fusion block;
- the behavior decoder;
- the two-channel behavior head.

They do not train the travel-time physics or BP decoder.

## V2.4 BP hazard repair

V2.4 converts upsampled physical BP into hazard:

$$
H_0=-\log(1-p^{phys}_{BP})
$$

The first BP head predicts a bounded local log multiplier:

$$
c(x)=\tanh(h_c(D_{BP}(x)))
$$

$$
H_c(x)=H_0(x)\exp[\log(1.5)c(x)]
$$

Thus the multiplicative range is approximately `[2/3, 1.5]`.

A second signed head predicts:

$$
r(x)=\tanh(h_r(D_{BP}(x)))b(x)
$$

Positive values add hazard:

$$
H_+(x)=0.002\max(r(x),0)
$$

Negative values attenuate existing hazard:

$$
a(x)=1-0.9\max(-r(x),0)
$$

The final hazard and BP are:

$$
H_{final}(x)=a(x)H_c(x)+H_+(x)
$$

$$
p_{BP}(x)=1-\exp[-H_{final}(x)]
$$

The model returns the logit of this probability.

This design allows v2.4 to:

- reduce overestimated physical BP;
- amplify underestimated nonzero physical BP;
- add a small positive hazard where physical BP is zero but fuel is burnable.

It also creates the main v2.4 limitation: the multiplier and signed residual
must serve very different correction regimes while operating within narrow
hard bounds.

## FI and ROS outputs

The behavior decoder is concatenated with three detached physical references:

$$
\log(1+{\rm HFI}^{phys}),\quad
\log(1+{\rm ROS}^{phys}),\quad
\overline{\delta^{speed}}
$$

The behavior head emits FI and ROS in their standardized log target spaces.
Two consistency terms keep them related to the physical fields:

$$
{\cal L}_{FI,phys}
=
{\rm MSE}
\left(
\hat y_{FI},
\frac{\log(1+{\rm HFI}^{phys})-\mu_{FI}}{\sigma_{FI}}
\right)
$$

$$
{\cal L}_{ROS,phys}
=
{\rm MSE}
\left(
\sigma_{ROS}
\left[
\hat y_{ROS}
-
\frac{\log(1+{\rm ROS}^{phys})-\mu_{ROS}}{\sigma_{ROS}}
\right],
\overline{\delta^{speed}}
\right)
$$

The reference weight on the combined behavior consistency term is `0.02`.

## Training objective

The supervised task objective is:

$$
{\cal L}_{task}
=
0.50{\cal L}_{BP}
+0.25{\cal L}_{FI}
+0.25{\cal L}_{ROS}
$$

BP combines:

- Bernoulli KL: `0.45`;
- CCC: `0.45`;
- hex-mean pairwise rank: `0.05`;
- hex-top-10 pairwise rank: `0.05`.

FI and ROS each combine:

- Huber: `0.70`;
- regression Pearson: `0.30`.

Additional terms are:

- q10/q50/q90 weight KL regularization;
- FI/ROS physical consistency;
- coarse BP KL/CCC supervision, weight `0.05`;
- signed BP residual L2 regularization, weight `0.01`.

The effective batch size is:

$$
4\ {\rm patches/batch}\times16\ {\rm accumulation\ steps}=64
$$

The reference schedule uses AdamW at `5e-4`, five warmup epochs, cosine decay,
gradient clipping at 1.0, and 40 epochs.

## Checkpointing and model selection

`Trainer.save_model()` stores:

- model, optimizer, and scheduler state;
- complete validated config;
- epoch and global step;
- current and best metric values;
- Python, NumPy, CPU Torch, and CUDA RNG states;
- Comet experiment key when applicable.

Writes are atomic:

```text
torch.save(..., path.tmp)
os.replace(path.tmp, path)
```

The reference checkpoint is selected by stitched validation mean CCC across
BP, FI, and ROS, not by patch loss.

## Evaluation and inference

### National stitched evaluation

`run_files/mechanistic_hybrid_v24/evaluate.sh`:

1. creates an output directory;
2. safely symlinks the selected checkpoint as `best.pth`;
3. creates a temporary evaluation config;
4. invokes `python -m src.evaluate_hexels`;
5. reconstructs each target on the native hexel grid;
6. writes metrics, rasters, and optional plots.

### Tensor-level inference

`inference/predictor.py::BurnRiskPredictor`:

1. reads the checkpoint config;
2. rebuilds the exact model through `build_model`;
3. strictly loads `model_state`;
4. predicts full-resolution fields;
5. center-crops to the configured target;
6. applies sigmoid to BP and identity to FI/ROS;
7. returns CPU tensors.

At this layer FI and ROS remain in standardized log target space. Stitched
evaluation and regional reconstruction apply the configured inverse
`log_standard` transform before metrics and output artifacts.

### Regional end-to-end inference

`inference/run_ai_surrogate_model_hexel_inference.py`:

1. validates or prepares regional inputs;
2. reuses normalization and fuel artifacts from `training_data_root`;
3. constructs the same source classes as training;
4. predicts patches;
5. stitches BP/FI/ROS to the native grid;
6. saves per-target GeoTIFFs;
7. computes raw hazard:

$$
{\rm Hazard}
=
{\rm BP}\times\min({\rm FI},10000)
$$

The current workflow still requires BP/FI/ROS reference rasters during
preparation and plotting. Feature-only regional inference is not yet a
supported release path.

## Reference performance and observed limitations

| Metric | V2.4 |
| --- | ---: |
| Validation BP/FI/ROS CCC at selected checkpoint | 0.8256 / 0.7617 / 0.7298 |
| Stitched test BP CCC | 0.819621 |
| Stitched test FI CCC | 0.782212 |
| Stitched test ROS CCC | 0.711031 |
| Stitched test mean CCC | 0.770954 |
| Test BP/FI/ROS top-10 IoU | 0.520382 / 0.336043 / 0.286883 |

The model is the strongest overall hybrid so far, but its diagnostics expose
specific next-step problems:

- BP log-hazard multiplication was approximately 99.4% saturated at the best
  epoch.
- The signed BP residual was approximately 54% saturated and predominantly
  negative.
- BP ranking/localization is strong (`Spearman = 0.9326`), but BP MAE
  (`0.001385`) is worse than v2.3 (`0.001190`).
- BP regressed relative to v2.3 on hex01 and hex16 despite large gains on
  other regions.
- FI retained a positive physical-unit bias of approximately `+82 kW/m`.
- Speed correction and propagation-cap diagnostics were healthy; FI/ROS do
  not show the same saturation pathology as BP.

These observations motivate a BP-only v2.5 correction redesign followed by a
separate FI/ROS calibration-only v2.6.
