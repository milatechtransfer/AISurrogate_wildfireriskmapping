# Mechanistic Fire Propagation Model

## Purpose

The objective is to predict three Burn-P3 outputs:

- burn probability (BP);
- conditional fire intensity (FI);
- conditional rate of spread (ROS).

A standard U-Net can learn these outputs directly, but it is free to use any statistical shortcut. The mechanistic model instead introduces an explicit intermediate representation:

1. ignition creates low-probability source cells;
2. fire moves through eight directional transmission fields;
3. non-burnable fuel blocks movement;
4. the fire-size distribution progressively stops long-distance spread;
5. accumulated reachability determines BP;
6. decoded local features determine FI and ROS.

The model is still a surrogate, not a complete physical fire simulator. Its propagation equations are differentiable so all components can be trained end-to-end.

## Current status

Two versions should be distinguished.

### Version 1: completed experiment

The first run reached validation CCC:

| Target | CCC |
|---|---:|
| BP | 0.559 |
| FI | 0.544 |
| ROS | 0.534 |

It stopped after epoch 20 because the 16-hour allocation expired.

Diagnostics showed that version 1 did not use propagation as intended:

- 98.7% of input pixels had positive ignition intensity;
- 94.7% of fully burnable coarse cells received initial seed probability above 0.9;
- central reachability was commonly between 0.97 and 1.00;
- propagation added only approximately 1-5% of final reachability;
- a flexible BP calibration head therefore performed most of the BP prediction.

The problem was not simply insufficient training. Dense ignition-rate rasters were treated like sparse ignition events and were summed within each coarse cell, which saturated the initial probability.

### Version 2: submitted on the corrected `512 -> 256` geometry

Version 2 changes the ignition and BP calibration equations so propagation cannot be bypassed:

- ignition intensity is mean-pooled rather than summed;
- ignition scale is positive and bounded;
- burnability is applied once per propagation stage rather than repeatedly before and inside propagation;
- BP uses a global scale plus a bounded local correction instead of an unrestricted spatial calibration gate;
- encoder width is increased from 24 to 48 base channels.

The approved version 2 run uses q10, q50, and q90 on the correctly tiled `512 -> 256` dataset. Its training job is `10393563`.

## Input representation

The approved version 2 experiment has 20 spatial channels:

| Source | Channels |
|---|---:|
| Human and lightning ignition | 2 |
| Elevation, slope, aspect sine, aspect cosine | 4 |
| Spatialized weather | 11 |
| Fire-size q10, q50, q90 | 3 |
| **Total** | **20** |

Fuel behavior is supplied separately as an 18-channel iROS curve at every pixel. A small fuel-curve encoder converts this to a learned embedding before concatenation with the spatial inputs.

The full input is `512 x 512` pixels at 100 m resolution. The propagation grid is downsampled by eight:

```text
512 x 512 source pixels
        |
        v
64 x 64 propagation cells
```

One propagation cell therefore represents an `800 m x 800 m` area.

## Neural encoder

Let:

- \(X\) be the 20 spatial input channels;
- \(C\) be the per-pixel 18-channel iROS curve;
- \(E_f(C)\) be the learned fuel-curve embedding.

The local encoder receives:

$$
Z_0 = [X, E_f(C)].
$$

Version 2 uses feature widths:

```text
full resolution: 48 channels
1/2 resolution:  96 channels
1/4 resolution: 192 channels
1/8 resolution: 288 channels
```

The `1/8` features are used to predict directional transmission. A U-Net-style decoder later combines propagation results with higher-resolution skip features.

## Ignition seeding

Human and lightning ignition surfaces are added:

$$
I(x) = I_{\mathrm{human}}(x) + I_{\mathrm{lightning}}(x).
$$

These maps are dense relative-rate surfaces, not observed individual ignition points.

### Version 1 equation

Version 1 averaged the source pixels and then multiplied by the number of pixels in an `8 x 8` propagation cell. This is equivalent to summing:

$$
I_{\mathrm{sum}}(u)
  = 64 \cdot \operatorname{AvgPool}_{8 \times 8}(I)(u).
$$

It converted the sum to a Bernoulli probability:

$$
p_{\mathrm{seed}}(u)
  = 1 - \exp(-\alpha I_{\mathrm{sum}}(u)).
$$

For a typical pooled sum near 11 and learned \(\alpha \approx 0.35\), this gives:

$$
1 - \exp(-0.35 \cdot 11) \approx 0.98.
$$

Almost every burnable cell therefore started as already reached.

### Version 2 equation

Version 2 retains the mean ignition rate:

$$
\bar I(u)
  = \operatorname{AvgPool}_{8 \times 8}(I)(u).
$$

The scale is bounded:

$$
\alpha = \alpha_{\max}\,\sigma(a),
$$

where:

- \(\alpha_{\max}=1\);
- the initial value is \(\alpha=0.05\);
- \(a\) is learned.

The seed probability is:

$$
p_{\mathrm{seed}}(u)
  = 1 - \exp(-\alpha \bar I(u)).
$$

For dense ignition values of `0.2 + 0.2`, the initial seed is:

$$
1 - \exp(-0.05 \cdot 0.4) \approx 0.0198.
$$

This leaves room for directional propagation to explain where BP becomes large.

## Burnability from iROS

At source resolution, a pixel is burnable if at least one iROS channel is positive:

$$
B_{\mathrm{fine}}(x)
  = \mathbf{1}\left[\sum_c \max(C_c(x),0) > 0\right].
$$

The propagation-cell burnability is the fraction of burnable source pixels:

$$
B(u)
  = \operatorname{AvgPool}_{8 \times 8}(B_{\mathrm{fine}})(u).
$$

Therefore:

- \(B=0\) is a complete barrier;
- \(B=1\) is fully burnable;
- \(0 < B < 1\) is a mixed coarse cell.

Version 2 applies this factor when initializing reachability and when a new frontier enters a destination cell. It does not pre-multiply ignition and transmission by the same fractional factor, avoiding repeated powers of \(B\).

## Directional transmission

The coarse neural features predict eight transmission probabilities:

```text
N, NE, E, SE, S, SW, W, NW
```

For direction \(d\):

$$
T_d(u) = \sigma(g_d(Z(u))),
$$

where \(g_d\) is a learned convolution and \(T_d \in [0,1]\).

The transmission fields can use:

- wind direction and strength;
- fuel behavior encoded by iROS;
- slope and aspect;
- weather;
- surrounding spatial context.

The equations impose directional movement, but the neural network learns how the input variables control each direction.

## Differentiable propagation

Let:

- \(R_t(u)\) be probability that cell \(u\) has been reached by step \(t\);
- \(F_t(u)\) be the newly reached frontier at step \(t\);
- \(\delta_d\) be the offset for direction \(d\).

Initialization is:

$$
R_0(u) = p_{\mathrm{seed}}(u)B(u),
$$

$$
F_0(u) = R_0(u).
$$

### Directional messages

A source cell sends:

$$
M_{t,d}(u)
  = F_{t-1}(u-\delta_d)T_d(u-\delta_d).
$$

### Noisy-OR arrival

Several neighbors may reach the same destination. They are combined with a noisy-OR:

$$
A_t(u)
  = 1 - \prod_d \left(1-M_{t,d}(u)\right).
$$

This is differentiable and avoids a hard threshold.

### New frontier

The new frontier is:

$$
F_t(u)
  = \left(1-R_{t-1}(u)\right)
    A_t(u)
    C_t(u)
    B(u),
$$

where \(C_t\) is the fire-size continuation probability described below.

Reachability is updated by:

$$
R_t(u) = R_{t-1}(u) + F_t(u).
$$

The factor \(1-R_{t-1}\) prevents repeatedly adding probability to cells that are already reached.

The configured model uses 24 propagation steps. At 800 m per step, the nominal cardinal-direction radius is:

$$
24 \times 800\ \mathrm{m} = 19.2\ \mathrm{km}.
$$

Diagonal and cardinal moves currently count as one step even though a diagonal move is geometrically longer. This is an approximation.

## Fire-size distribution

### Source data

The fire-size table contains:

- 32,120 individual fire-size observations;
- 60 `GRIDCODE` fire zones;
- between 1 and 3,350 observations per zone;
- a median of 381 observations per zone.

Each raw fire size is transformed as:

$$
L = \log_{10}(\mathrm{SIZE}_{ha}+1).
$$

This is base-10 logarithm, not natural logarithm.

### Zone versus season indexing

The current fire-size table is indexed only by `GRIDCODE`.

It does not contain:

- season;
- ignition cause;
- hex ID;
- simulation year.

All fire observations for a zone are therefore pooled into one zone-level distribution. The same fire-size channels are used for every season and every patch intersecting that zone.

A seasonal distribution would require:

1. a season column in the fire-size table;
2. a lookup keyed by `(GRIDCODE, season)` rather than only `GRIDCODE`;
3. the patch or sample season to be available when rasterizing the channels.

That behavior is not implemented in the current data source.

### Quantiles available and selected

The original prototype used nine values for every fire zone:

```text
q10, q20, q30, q40, q50, q60, q70, q80, q90
```

Equivalently, the configured quantiles are:

```yaml
quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

For zone \(z\), channel \(q\) is:

$$
L_q(z)
  = \operatorname{Quantile}_q
    \left(\log_{10}(\mathrm{SIZE}_{ha}+1)\right).
$$

The approved version 2 experiment instead selects:

```text
q10, q50, q90
```

All selected channels are supplied simultaneously. The model does not randomly select one percentile for a training example.

Every pixel in the same fire zone receives the same selected values. If a raster zone is absent from the table, the implementation uses global quantiles from all rows.

The global fallback values are approximately:

| Quantile | `LOG_SIZE_HA` | Equivalent hectares |
|---|---:|---:|
| q10 | 1.576 | 36.7 |
| q20 | 1.732 | 53.0 |
| q30 | 1.919 | 82.0 |
| q40 | 2.149 | 140.0 |
| q50 | 2.400 | 250.0 |
| q60 | 2.684 | 481.9 |
| q70 | 3.000 | 1,000.0 |
| q80 | 3.386 | 2,431.9 |
| q90 | 3.894 | 7,837.8 |

These are fallback values, not values shared by all zones.

Across zones, the median zone-level values are approximately:

| Quantile | Median hectares |
|---|---:|
| q10 | 39.6 |
| q50 | 220.0 |
| q90 | 4,325.0 |

Zones with very few observations have uncertain quantiles. A zone with one observation has the same value for all nine percentiles.

### How many quantile channels should be used?

The number and positions of the quantiles are configurable. Nine deciles are the current configuration, not a hard architectural requirement of the data source.

Nine channels do not increase the stored context-patch dataset size because the channels are rasterized from the small CSV at loading time. They do increase:

- input tensor memory;
- first-layer computation;
- the number of strongly correlated input features.

At `512 x 512` float32 resolution:

- nine channels occupy approximately 9 MiB per sample;
- three channels occupy approximately 3 MiB per sample.

This raw-input difference is small compared with the encoder activations, but nine zone-constant channels may be statistically redundant.

A compact and interpretable distribution is:

```yaml
quantiles: [0.1, 0.5, 0.9]
```

This gives:

- q10: representative small-fire scale;
- q50: median fire scale;
- q90: representative large-fire tail.

A four-channel alternative is:

```yaml
quantiles: [0.1, 0.35, 0.65, 0.9]
```

The spatialized-tabular source accepts any sorted, unique list strictly between zero and one.

### Mean-only fire-size baseline

The original fire-size baseline behavior is also available. Omitting `quantiles` and retaining mean aggregation produces one channel:

```yaml
- name: "spatialized_fire_size"
  params:
    csv_name: "df_fire_fru_processed.csv"
    feature_names_list: ["NORM_LOG_SIZE_HA"]
    fire_weather_zone_id_col: "GRIDCODE"
    aggregation: "mean"
    zone_channel_key: "firezones_grid"
```

This is appropriate for a standard U-Net ablation because the network treats fire size as an ordinary covariate.

For a mechanistic stopping equation, the physical raw-log feature should be used:

```yaml
feature_names_list: ["LOG_SIZE_HA"]
aggregation: "mean"
```

A mean-only mechanistic model interprets this as a single representative fire size rather than a distribution. The implementation now supports this mode, although the approved mechanistic run uses q10, q50, and q90.

### Required area at each propagation step

At step \(t\), the model approximates the area required to reach radius \(r_t\):

$$
r_t = t \cdot 800\ \mathrm{m},
$$

$$
A_t
  = m_A \frac{\pi r_t^2}{10,000},
$$

where:

- \(A_t\) is in hectares;
- \(m_A\) is a learned area multiplier bounded between 0.05 and 20.

The size threshold is transformed to the same log scale:

$$
\ell_t = \log_{10}(A_t+1).
$$

### Smoothed survival probability

The model estimates the probability that a fire is large enough to reach step \(t\):

$$
S_t(u)
  = \sum_i w_i
    \sigma\left(k\left[L_{q_i}(u)-\ell_t\right]\right),
$$

with sharpness \(k=6\).

The weights are derived from the spacing between configured percentile levels. For q10, q50, and q90:

$$
[w_{10},w_{50},w_{90}] = [0.3,0.4,0.3].
$$

Interpretation:

- a quantile far above the required size contributes nearly 1;
- a quantile far below it contributes nearly 0;
- the weighted sum approximates the fraction of the fire-size distribution that exceeds the required area.

This is a compact approximation to an empirical survival curve. It does not explicitly represent the distribution below q10 or above q90.

### Conditional continuation

The continuation applied at step \(t\) is:

$$
C_t(u)
  = \operatorname{clip}
    \left(
      \frac{S_t(u)}
           {\max(S_{t-1}(u),\epsilon)},
      0,
      1
    \right).
$$

This conditional ratio is important. The product through step \(t\) becomes approximately:

$$
\prod_{j=1}^{t} C_j \approx S_t.
$$

Multiplying by the cumulative survival \(S_t\) at every step would stop fires much too aggressively.

## Burn-probability calibration in version 2

After propagation, coarse reachability is bilinearly upsampled to the original `512 x 512` resolution.

Version 1 multiplied reachability by an unrestricted spatial sigmoid head. When reachability saturated near one, this head could predict BP almost independently.

Version 2 uses:

1. a positive global BP scale \(\beta\);
2. a bounded local log correction.

The local correction is:

$$
\ell_{\mathrm{local}}(x)
  = \log(2)\tanh(h(Z_{\mathrm{decoded}}(x))).
$$

Therefore:

$$
\exp(\ell_{\mathrm{local}}) \in [0.5,2].
$$

The BP hazard is:

$$
H_{\mathrm{BP}}(x)
  = \beta R(x)\exp(\ell_{\mathrm{local}}(x)).
$$

The final physical BP is:

$$
\widehat{\mathrm{BP}}(x)
  = 1-\exp(-H_{\mathrm{BP}}(x)).
$$

The local decoder can adjust BP by at most a factor of two. It cannot replace reachability with an arbitrary prediction map.

## FI and ROS heads

FI and ROS are predicted from the decoded neural features after propagation features have been fused:

$$
[\widehat{\mathrm{FI}},\widehat{\mathrm{ROS}}]
  = h_{\mathrm{behavior}}(Z_{\mathrm{decoded}}).
$$

The model therefore imposes the strongest structure on BP. FI and ROS retain more flexible local prediction heads because they depend on local fuel and weather behavior conditional on burning.

## Training targets and losses

Only the centered output crop contributes to losses and metrics.

### BP

BP is trained in physical probability units rather than min-max stretching.

The configured BP loss combines:

- Bernoulli KL: 45%;
- CCC loss: 45%;
- hex mean pairwise ranking: 5%;
- hex top-10% pairwise ranking: 5%.

BP receives 50% of the total multi-task weight.

### FI and ROS

FI and ROS are log-standardized and each uses:

- Huber loss: 70%;
- raw Pearson loss: 30%.

Each receives 25% of the total multi-task weight.

### Batch geometry

The mechanistic experiment uses:

- physical batch size: 4;
- gradient accumulation: 16;
- effective batch size: 64;
- planned epochs: 25.

## `512 -> 256` and `512 -> 128` geometry

The overlap parameter is defined on the supervised target crop, not directly on the input window.

The preparation code uses:

$$
\mathrm{stride}
  = \operatorname{int}
    \left(
      \mathrm{target\ crop}
      \cdot (1-\mathrm{overlap})
    \right).
$$

### Standard `256 -> 256`

With a 256-pixel target and configured overlap 0.2:

$$
\mathrm{stride} = \operatorname{int}(256 \cdot 0.8)=204.
$$

The realized output overlap is:

$$
\frac{256-204}{256}=20.31\%.
$$

### Generated `512 -> 256`

The 512 input has 128 pixels of context on each side:

```text
|-- 128 context --|------ 256 target ------|-- 128 context --|
```

The target stride remains 204 pixels, so target overlap is approximately 20%.

The input windows overlap by:

$$
\frac{512-204}{512}=60.16\%.
$$

Therefore `overlap_ratio: 0.2` does not mean 20% overlap between the 512-pixel inputs. It means approximately 20% overlap between the 256-pixel prediction tiles.

### Current lightweight `512 -> 128` view

The current `512 -> 128` dataset was not regenerated with new patch centers. It reuses the existing `512 -> 256` arrays and keeps their centers.

It changes:

- target crop from 256 to 128;
- target top-left row and column by `+64`;
- context from 128 to 192 pixels per side.

```text
|------ 192 context ------|-- 128 target --|------ 192 context ------|
```

The stride remains 204 pixels because the centers come from the `512 -> 256` dataset.

Consequences:

- 512-pixel input overlap remains approximately 60%;
- 128-pixel targets do not overlap;
- neighboring 128-pixel targets have a `204-128=76` pixel gap;
- patch-level training and validation are valid;
- gap-free full-raster reconstruction is not valid with this lightweight view.
- each epoch supervises approximately one quarter as many output pixels as a true dense `512 -> 128` tiling over the same area;
- the retained `valid_ratio` was measured on the broader 256-pixel crop, so it is only an approximation for the centered 128-pixel crop.

Therefore the lightweight view should be treated as a quick patch-level diagnostic, not as the final geometry for a controlled production experiment.

To generate a true `512 -> 128` dataset with 20% target overlap:

$$
\mathrm{stride}
  = \operatorname{int}(128 \cdot 0.8)
  = 102.
$$

This would give:

- approximately 20% overlap between 128-pixel targets;
- approximately 80% overlap between 512-pixel inputs;
- roughly four times as many patch centers because stride is halved in both dimensions.

For the next controlled experiments, the practical choices are:

1. use the already generated, correctly tiled `512 -> 256` dataset; or
2. regenerate a true `512 -> 128` dataset with stride 102.

The first option is substantially cheaper and is the cleaner immediate comparison.

## Previous larger-context experiment

We previously trained U-Net at:

- `256 -> 256`;
- `512 -> 256`.

However, both context-matrix configurations extended:

```yaml
multi_output_spatial_weather_firesize.yaml
```

Therefore both included a spatialized fire-size input.

The corrected validation results were:

| Model | BP CCC | FI CCC | ROS CCC | Hazard CCC |
|---|---:|---:|---:|---:|
| U-Net `256 -> 256` with fire size | 0.594 | 0.586 | 0.562 | 0.618 |
| U-Net `512 -> 256` with fire size | 0.601 | 0.595 | 0.569 | 0.633 |
| Difference | +0.007 | +0.009 | +0.007 | +0.015 |

This shows a small benefit from larger context when fire size is present.

It does not answer the clean question:

> Does larger context help the current no-fire-size multi-output model?

The current paper/reference checkpoint uses:

- `256 -> 256`;
- 20% output-tile overlap;
- spatial grids and spatialized weather;
- no fire-size input.

A clean no-fire-size context experiment still needs:

- the same `multi_output_spatial_weather.yaml` inputs and losses;
- the existing `512 -> 256` context dataset;
- no spatialized fire-size source;
- identical effective batch size;
- comparison against the existing `256 -> 256` no-fire-size checkpoint.

That experiment would isolate context size without changing the fire-size information available to the model.

## Controls for evaluating the mechanistic model

The submitted strong U-Net control for version 2 uses exactly the same:

- correctly tiled `512 -> 256` patches;
- 20 spatial channels;
- q10, q50, and q90 fire-size channels;
- iROS fuel curves;
- physical BP target;
- BP/FI/ROS losses;
- optimizer schedule;
- effective batch size.

The only intended difference is:

- mechanistic version 2: explicit ignition, transmission, stopping, and reachability;
- control: standard U-Net direct prediction.

Parameter counts are:

| Model | Parameters |
|---|---:|
| Mechanistic version 2 | 5.0 million |
| Strong U-Net control | 34.5 million |

This is a competitiveness control, not a parameter-matched control.

Interpretation:

| Result | Meaning |
|---|---|
| Both models poor | The crop, physical-BP objective, or input setup is likely the limitation |
| U-Net strong, mechanism poor | The mechanistic constraints are harmful or incorrectly specified |
| Mechanism near U-Net | The structure is competitive with substantially fewer parameters |
| Mechanism better | The propagation inductive bias is helping |

## Main limitations

1. Ignition inputs are relative rate surfaces, not observed ignition events.
2. Fire-size distributions are zone-level and constant within each zone.
3. Nine deciles approximate, but do not fully preserve, the empirical distribution.
4. Zones with few observations have unstable quantiles.
5. Propagation is performed on 800 m cells, which removes fine-scale barriers.
6. Cardinal and diagonal moves currently consume the same distance step.
7. The noisy-OR assumes approximate independence between incoming paths.
8. FI and ROS remain mostly neural rather than explicitly mechanistic.
9. The historical lightweight `512 -> 128` view cannot reconstruct a gap-free raster and is not used by the approved runs.

## Relevant implementation files

- `src/models/mechanistic_propagation.py`
- `configs/mechanistic/mechanistic_propagation_512_crop_128_firesize_quantiles.yaml`
- `configs/mechanistic/mechanistic_propagation_v2_512_crop_256_firesize_q3.yaml`
- `configs/mechanistic/unet_512_crop_256_firesize_q3_physical_control.yaml`
- `configs/context_models/unet_512_crop_256_no_firesize.yaml`
- `configs/context_models/unet_512_crop_256_firesize_q3.yaml`
- `src/datasets/sources/spatialized_tabular.py`
- `data_preparation/prepare_context_crop_view.py`
- `data_preparation/process_hexels_into_grids.py`
