# Mechanistic hybrid v2.5 design: BP correction without saturation

## Purpose

V2.5 is a controlled **BP-only correction redesign** of v2.4.

It is not a new propagation model and it is not the FI/ROS experiment.
Relative to a geometry-matched v2.4 control, v2.5 should keep all of the
following unchanged:

- shared encoder;
- iROS/HFI curve encoders;
- source-cohort travel-time propagation;
- ignition-count distribution;
- direct log-hectare q10/q50/q90 fire-size model;
- behavior fusion, decoder, and FI/ROS head;
- supervised BP/FI/ROS task losses and task weights;
- stitched validation checkpoint criterion.

The only architectural change is the BP hazard correction after physical BP.

## Why v2.5 is needed

V2.4 achieved the best overall hybrid result:

| Test metric | V2.2 | V2.3 | V2.4 |
| --- | ---: | ---: | ---: |
| BP CCC | 0.6735 | 0.7954 | **0.8196** |
| FI CCC | **0.7873** | 0.7456 | 0.7822 |
| ROS CCC | **0.7165** | 0.6791 | 0.7110 |
| Mean CCC | 0.7258 | 0.7400 | **0.7710** |

The remaining BP problem is calibration capacity, not localization:

- BP Spearman is `0.9326`;
- BP MAE is `0.001385`, worse than v2.3's `0.001190`;
- the local log-hazard multiplier is approximately **99.4% saturated**;
- the signed hazard residual is approximately **54% saturated**;
- the signed residual is predominantly negative, with mean approximately
  `-0.712`;
- BP improved nationally but regressed on hex01 and hex16.

The current v2.4 correction is:

$$
H_c
=
H_0\exp[L\tanh(z_c)]
$$

$$
H_{final}
=
\left[1-A\max(-\tanh(z_r),0)\right]H_c
+R\max(\tanh(z_r),0)
$$

with:

- $L=\log(1.5)$;
- $A=0.9$;
- $R=0.002$.

This asks two narrow `tanh` fields to cover three distinct jobs:

1. attenuate overpredicted nonzero physical hazard;
2. amplify underpredicted nonzero physical hazard;
3. create positive hazard where physical hazard is zero.

The observed saturation shows that these jobs should no longer share the same
bounded parameterization.

## Geometry recommendation: native 256x256

The geometry decision is independent of the BP-head design, but it should be
resolved while creating v2.5.

The matched q3 U-Net multiseed experiment used seeds 42, 1337, and 2024:

| Held-out test CCC, three-seed mean | 256->256 | 512->256 | 512 minus 256 |
| --- | ---: | ---: | ---: |
| Patch BP | 0.694142 | 0.693123 | -0.001019 |
| Patch FI | 0.683189 | 0.664923 | -0.018266 |
| Patch ROS | 0.637438 | 0.626609 | -0.010829 |
| Patch hazard | 0.729970 | 0.728158 | -0.001812 |
| Stitched BP | 0.854080 | 0.849930 | -0.004151 |
| Stitched FI | 0.790133 | 0.772932 | -0.017202 |
| Stitched ROS | 0.716984 | 0.705076 | -0.011908 |

On validation, 512 context improved BP, but the gain did not generalize to the
held-out test set and FI/ROS were lower. Mean training epoch time was:

| Geometry | Mean epoch time |
| --- | ---: |
| 256->256 | 1130.9 s |
| 512->256 | 2909.6 s |

Thus 512 used four times the input area and took **2.57x** longer per epoch
without a held-out-test gain.

This is U-Net evidence rather than a direct mechanistic ablation. The
recommended v2.5 experiment therefore separates geometry from the BP change:

1. train a native-256 **v2.4 geometry control**;
2. train native-256 v2.5 with the same seed and schedule;
3. keep 512->256 only as an optional diagnostic control.

The default v2.5 design target is native 256x256.

## Proposed BP correction

Let:

- $H_0=-\log(1-p^{phys}_{BP})$: physical hazard;
- $D_{BP}(x)$: v2.4 BP decoder feature map;
- $b(x)$: fine-resolution burnability mask.

### 1. Separate attenuation and amplification

Use two non-negative fields:

$$
d_+(x)=s_+\,{\rm softplus}(h_+(D_{BP}(x)))
$$

$$
d_-(x)=s_-\,{\rm softplus}(h_-(D_{BP}(x)))
$$

The biases must be chosen so:

$$
d_+(x)-d_-(x)\approx0
$$

at initialization. The calibrated nonzero physical hazard is:

$$
H_{scaled}(x)
=
H_0(x)\exp[d_+(x)-d_-(x)]
$$

Interpretation:

- $d_+$ only amplifies;
- $d_-$ only attenuates;
- their scales and regularization can be asymmetric;
- neither relies on a narrow `tanh` bound.

V2.4 diagnostics show attenuation needs substantially more capacity than
amplification. The initial v2.5 configuration should therefore permit a
larger attenuation scale or apply a weaker attenuation penalty, while still
logging both paths independently.

### 2. Separate positive-support gate

Multiplication cannot repair:

$$
H_0(x)=0
$$

because any finite multiplier leaves it zero. V2.5 should use a separate
positive-hazard path only where physical support is small.

Define a deterministic near-zero physical-support gate:

$$
g_0(x)
=
\sigma\left(
\frac{H_{gate}-H_0(x)}{\tau_{gate}}
\right)
$$

and a learned content gate:

$$
g_n(x)=\sigma(h_g(D_{BP}(x)))
$$

The non-negative support magnitude is:

$$
H_{support}(x)
=
b(x)\,
g_0(x)\,
g_n(x)\,
s_h{\rm softplus}(h_h(D_{BP}(x)))
$$

The final BP model is:

$$
H_{final}(x)
=
H_{scaled}(x)+H_{support}(x)
$$

$$
p_{BP}(x)
=
1-\exp[-H_{final}(x)]
$$

All correction outputs, diagnostics, and penalties are evaluated on burnable
support. Existing physical hazard is already zero outside that support, and
the new positive-support path retains the explicit $b(x)$ mask.

This cleanly separates:

- correction of existing physical support;
- creation of missing physical support.

The support magnitude should initialize near v2.4's `1e-4` initial hazard so
the first forward pass is conservative.

### 3. Soft regularization, not narrow learned caps

Keep the supervised BP loss unchanged. Replace the current signed-residual L2
term with correction-specific soft penalties:

$$
{\cal L}_{corr}
=
\lambda_m\,{\mathbb E}_b[d_+ + d_-]
+\lambda_c\,{\mathbb E}_b[d_+d_-]
+\lambda_h\,{\mathbb E}_b[H_{support}]
$$

The co-activation term discourages simultaneous attenuation and amplification
at one pixel. Numerical safety clamps may exist far outside the expected
operating range, but they must not act as the normal learned capacity limit.

## Gradient contract

The v2.4 task separation must remain:

| Loss | May update |
| --- | --- |
| BP | Shared encoder, physical correction fields/scalars, BP fusion/decoder, v2.5 BP heads |
| FI/ROS | Shared encoder, behavior fusion/decoder/head |
| FI/ROS | Must not update travel-time physics, BP decoder, or v2.5 BP heads |

No behavior module should be added or widened in v2.5.

## Native-256 implementation implications

### Data

Create a separate immutable 256x256 prepared-data root. Do not mutate the v2.4
512->256 dataset. The new root should preserve all v2.4 feature semantics:

- native aligned 100 m cells;
- exact human/lightning probability mass;
- direct log-hectare q10/q50/q90 plus missing mask;
- ignition-count mean and CV;
- frozen weather/count/fire-size artifacts;
- iROS/HFI fuel curves.

### Configuration

The native control and v2.5 config should use:

```yaml
data_prep:
  win_h: 256
  win_w: 256
  target_crop_h: null
  target_crop_w: null
  preserve_native_grid: true
```

Batch size should be benchmarked while keeping effective batch size 64.

### Geometry validation

The current trainer requires:

$$
{\rm propagation\ steps}
\le
\frac{{\rm context\ margin}}{8}
$$

which rejects native 256 because its explicit margin is zero. This check must
be versioned for v2.5. The travel-time solver itself works on the 32x32 coarse
grid; 16 steps remain a valid path-length cap inside that tile.

The replacement validation should require:

- input dimensions divisible by eight;
- propagation steps no larger than a documented coarse-grid computational
  limit;
- explicit reporting of travel-time cap-hit rate;
- a boundary-sensitivity test because external ignitions are unavailable at
  native tile edges.

### Stitching and inference

Native 256 disables the configured center-cropping path in:

- `src/trainer.py`;
- `inference/predictor.py`;
- stitched reconstruction metadata.

The shared crop feature should remain intact for old checkpoints and the
optional 512 control. Existing native-grid reconstruction supports equal
input/output geometry, but regression tests must cover row/column placement,
overlap, and edge padding.

## Exact experiment matrix

### Experiment A: geometry control

`mechanistic_hybrid_v24_native_256_control`

- v2.4 model class unchanged;
- native 256 input/output;
- propagation and losses unchanged;
- seed 42;
- train from scratch;
- stop by 30-32 epochs unless validation is still improving.

This measures geometry alone.

### Experiment B: v2.5

`mechanistic_hybrid_v25_native_256`

- same geometry, seed, initialization policy, data, and schedule as A;
- only BP correction changes;
- all FI/ROS modules and losses unchanged.

This measures the BP redesign.

### Optional experiment C

`mechanistic_hybrid_v25_native_512_crop_256`

Run only if A suggests that the mechanistic model, unlike the q3 U-Net,
materially needs the halo. It is not the default release candidate.

## Implementation map

| File | Required change |
| --- | --- |
| `src/models/mechanistic_hybrid_v25.py` | Subclass v2.4, replace `bp_local_calibration` and `bp_hazard_residual` with separate attenuation, amplification, support-gate, and support-magnitude heads. |
| `src/models/factory.py` | Register `mechanistic_hybrid_v25` aliases. |
| `src/config.py` | Add v2.5 correction scales, initialization, gate, and regularization fields; version the native-256 geometry validation. |
| `src/trainer.py` | Consume v2.5 correction diagnostics/regularization and version the geometry validation while preserving the existing no-crop path. |
| `data_preparation/process_hexels_into_grids.py` | Generate or verify the separate native-256 prepared root. |
| `configs/mechanistic_hybrid_v24_native_256_control.yaml` | Matched geometry control. |
| `configs/mechanistic_hybrid_v25_native_256.yaml` | Primary v2.5 recipe. |
| `run_files/mechanistic_hybrid_v25/` | Smoke, train, and stitched-evaluation launchers. |
| `inference/mechanistic_hybrid_v25.yaml` | Published native-256 inference contract. |
| `tests/test_mechanistic_hybrid_v25.py` | Head semantics, identity initialization, finite gradients, masks, and separation tests. |
| `tests/test_mechanistic_hybrid_v25_release.py` | Standalone config, shared artifacts, and strict checkpoint tests. |

## Required diagnostics

Report globally and by validation/test hex:

- mean and quantiles of $d_+$ and $d_-$;
- fraction of pixels using net amplification versus attenuation;
- multiplier percentiles;
- simultaneous $d_+>0$ and $d_->0$ fraction;
- support-gate activation;
- positive support hazard on zero-physical-BP pixels;
- positive support hazard on nonzero-physical-BP pixels;
- BP MAE, bias, Spearman, CCC, and top-10 IoU;
- BP metrics specifically on hex01 and hex16;
- propagation cap-hit rate;
- FI/ROS metrics as non-regression controls.

The old saturation metrics should remain for direct comparison where
possible, but v2.5 should emphasize correction magnitude distributions rather
than proximity to a narrow hard bound.

## Acceptance gates

V2.5 is accepted only if:

1. native-256 v2.5 improves BP over the matched native-256 v2.4 control;
2. stitched BP CCC and top-10 IoU match or exceed the v2.4 release target
   (`0.8196` and `0.5204`) or show a compelling multiseed improvement over the
   geometry control;
3. BP MAE improves from `0.001385`, with `0.001190` as the v2.3 reference;
4. hex01 and hex16 no longer show the same large regressions;
5. FI and ROS change by no more than approximately `0.005` CCC relative to
   the matched control;
6. correction diagnostics show useful headroom rather than another boundary
   collapse;
7. native 256 materially reduces training and inference cost;
8. all burnability, zero-count, monotonic fire-size, finite-gradient, and
   stitching invariants pass.

After the seed-42 architecture converges, confirm the selected design with at
least seeds 42, 1337, and 2024.

## Checkpoint compatibility

The shared encoder, physics, and behavior tensors can retain v2.4 shapes.
V2.4 checkpoints can therefore initialize common v2.5 components, while the
new BP heads load separately.

The primary scientific comparison should nevertheless train the matched
native-256 control and v2.5 from scratch. Warm-start compatibility is useful
for debugging and transfer, but it should not confound the architecture
comparison.
