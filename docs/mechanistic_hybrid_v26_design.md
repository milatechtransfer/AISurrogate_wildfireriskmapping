# Mechanistic hybrid v2.6 design: FI/ROS amplitude calibration

## Purpose

V2.6 is an **FI/ROS calibration-only** follow-up to an accepted v2.5.

It should not:

- redesign BP;
- change propagation;
- add another decoder;
- widen the shared encoder;
- change the selected v2.5 geometry;
- introduce a new fire-size or ignition representation.

The first v2.6 experiment changes the FI/ROS training objective and fine-tunes
only the existing behavior path. BP remains the accepted v2.5 result.

## Why FI/ROS need a separate stage

V2.4 recovered most of the behavior performance lost in v2.3:

| Stitched test CCC | V2.2 | V2.3 | V2.4 |
| --- | ---: | ---: | ---: |
| FI | **0.7873** | 0.7456 | 0.7822 |
| ROS | **0.7165** | 0.6791 | 0.7110 |

The remaining gaps to v2.2 are small:

$$
\Delta FI = -0.0051,\qquad
\Delta ROS = -0.0055
$$

FI also has a positive physical-unit bias of approximately `+82 kW/m`,
compared with approximately `-16 kW/m` in v2.3.

The v2.4 behavior loss is:

$$
{\cal L}_{behavior}
=
0.70{\cal L}_{Huber}
+0.30{\cal L}_{Pearson}
$$

Pearson rewards spatial association but is invariant to affine changes in
mean and scale. A prediction can therefore have high Pearson correlation
while remaining biased or under/over-dispersed.

V2.6 should directly optimize concordance without changing the behavior
architecture.

## Regression CCC

For valid prediction values $\hat y$ and targets $y$, concordance is:

$$
\rho_c
=
\frac{
2\,{\rm cov}(\hat y,y)
}{
{\rm var}(\hat y)
+{\rm var}(y)
+(\mu_{\hat y}-\mu_y)^2
}
$$

Unlike Pearson correlation, CCC penalizes:

- mean bias;
- variance mismatch;
- weak association.

The loss is:

$$
{\cal L}_{raw\_ccc}=1-\rho_c
$$

"Raw" means that the loss operates on the unactivated regression outputs. It
must not apply the BP sigmoid. For v2.6, prediction and target values remain
in the existing standardized `log1p(FI)` and `log1p(ROS)` training spaces.

The initial FI and ROS objective should be:

$$
{\cal L}_{FI}
=
0.65{\cal L}_{Huber}
+0.15{\cal L}_{raw\_Pearson}
+0.20{\cal L}_{raw\_CCC}
$$

$$
{\cal L}_{ROS}
=
0.65{\cal L}_{Huber}
+0.15{\cal L}_{raw\_Pearson}
+0.20{\cal L}_{raw\_CCC}
$$

This is the agreed first experiment. Do not add a learned calibration module
at the same time.

## Why behavior-only fine-tuning is necessary

In v2.4 and the proposed v2.5:

- mechanistic reference fields are detached before the behavior decoder;
- FI/ROS losses cannot update propagation or BP-specific decoders;
- FI/ROS losses can still update the shared encoder.

If the shared encoder remains trainable, an FI/ROS loss change can indirectly
change BP. That would violate the "calibration-only" goal.

The primary v2.6 run should therefore load the accepted v2.5 checkpoint and
freeze:

- ROS/HFI fuel-curve encoders;
- shared full/half/quarter/eighth CNN encoder;
- learned speed correction;
- propagation scalars and quantile weights;
- BP fusion and decoder;
- all v2.5 BP correction heads.

Train only:

- `behavior_mechanistic_fusion`;
- `behavior_decoder_blocks`;
- `behavior_head`.

BP loss and metrics may still be computed for monitoring, but they must have
no path to trainable parameters.

Retain the existing outer task weights (`0.50/0.25/0.25`) and the `0.02`
physical behavior-consistency term. Only the inner FI/ROS loss mixture changes
in the first v2.6 experiment.

## Geometry

V2.6 inherits the accepted v2.5 geometry unchanged. If v2.5 adopts native
256x256, v2.6 must remain native 256x256.

Do not reintroduce 512 context during the FI/ROS loss experiment. The paired
q3 multiseed evidence found lower held-out FI and ROS CCC with 512->256 and a
2.57x training-time cost. Changing geometry would confound the objective
comparison.

## Model and training contract

### Model

No new learned module is required. The first implementation can use the v2.5
model class directly. If release provenance requires an architecture string,
`mechanistic_hybrid_v26` should be a parameter-identical alias/subclass, not a
new decoder.

### Initialization

- start from the accepted v2.5 checkpoint;
- preserve every frozen tensor exactly;
- initialize no new model parameters;
- reset only optimizer and scheduler state for the behavior-only fine-tune.

### Schedule

Use a short calibration run:

- 5-10 epochs;
- behavior-only optimizer groups;
- conservative learning rate, initially `1e-4` or lower;
- gradient clipping at 1.0;
- no long warmup;
- early stop when stitched FI/ROS concordance stops improving.

The exact learning rate should be selected with one seed before multiseed
confirmation. Do not launch a broad architecture sweep.

### Checkpoint selection

Keep the existing stitched `hex/mean/ccc` checkpoint criterion. With BP
frozen at a constant value across calibration epochs:

$$
{\rm mean\ CCC}
=
\frac{CCC_{BP}^{fixed}+CCC_{FI}+CCC_{ROS}}{3}
$$

Maximizing this is exactly equivalent to maximizing the sum of FI and ROS
CCC, while preserving the v2.5 release criterion and avoiding a trainer
metric change. Report `(CCC_FI + CCC_ROS) / 2` separately for readability.

Since BP modules are frozen, any BP difference beyond numerical tolerance
indicates an implementation error.

## Implementation map

| File | Required change |
| --- | --- |
| `src/losses.py` | Add `RegressionCCCLoss`, equivalent to `CCCLoss(use_sigmoid=False)`, with masked finite-row behavior matching existing CCC. |
| `src/utils.py` | Register names such as `raw_ccc` and `regression_ccc` in `build_single_loss`. |
| `src/config.py` | Ensure the new loss name is accepted and add explicit behavior-only fine-tuning configuration if the existing freeze fields are insufficient. |
| `src/trainer.py` | Add explicit behavior-only freezing and optimizer grouping; keep stitched checkpoint selection unchanged. |
| `configs/mechanistic_hybrid_v26_native_256.yaml` | Point to the accepted v2.5 checkpoint and use the `0.65/0.15/0.20` FI/ROS loss mixture. |
| `run_files/mechanistic_hybrid_v26/` | Behavior-only smoke, train, and stitched-evaluation launchers. |
| `tests/test_losses.py` | Verify raw CCC identity, bias/scale sensitivity, masking, finite gradients, and no sigmoid. |
| `tests/test_mechanistic_hybrid_v26.py` | Verify only behavior modules receive gradients and BP output is bitwise or numerically unchanged. |
| `tests/test_configs.py` | Verify v2.6 differs from v2.5 only in initialization, freezing, schedule, and FI/ROS loss definitions, while retaining the checkpoint criterion. |

## Required loss tests

`RegressionCCCLoss` must demonstrate:

1. identical prediction and target produce zero loss;
2. adding a constant bias worsens loss even when Pearson remains one;
3. multiplying prediction variance worsens loss even when Pearson remains one;
4. masked pixels do not contribute;
5. rows with fewer than two valid values are ignored consistently;
6. gradients are finite;
7. no sigmoid is applied;
8. BP's existing sigmoid CCC behavior is unchanged.

## Required run diagnostics

Report for FI and ROS globally and by hex:

- CCC;
- Pearson/Spearman;
- MAE and normalized MAE;
- bias and normalized bias;
- prediction and target means;
- prediction and target standard deviations;
- standard-deviation ratio;
- least-squares slope and intercept;
- top-k IoU and full AUC-IoU;
- physical-reference consistency loss.

Also report:

- frozen v2.5 BP CCC, MAE, bias, and top-10 IoU;
- maximum absolute BP prediction difference before and after v2.6;
- a list of trainable parameter names;
- a list of frozen parameter groups.

## Experiment sequence

### Experiment A: loss-only calibration

- accepted v2.5 checkpoint;
- behavior modules trainable;
- `0.65 Huber / 0.15 Pearson / 0.20 raw CCC`;
- seed 42;
- 5-10 epochs.

This is the primary v2.6 experiment.

### Experiment B: weight sensitivity, only if needed

If A improves one behavior target but harms the other, compare only a small
set of CCC weights while keeping Huber dominant. For example:

```text
CCC weight: 0.10, 0.20, 0.30
Pearson weight: 0.25, 0.15, 0.05
Huber weight: 0.65
```

Do not change architecture in this sweep.

### Deferred fallback

Only if raw CCC fails to remove amplitude bias should a later experiment test
an identity-initialized global affine calibration:

$$
\hat y'_t=a_t\hat y_t+b_t,\qquad
a_t=1,\ b_t=0
$$

This is deliberately outside the first v2.6 implementation.

## Acceptance gates

V2.6 is accepted only if:

1. stitched test FI CCC exceeds the v2.2 reference `0.7873`;
2. stitched test ROS CCC exceeds the v2.2 reference `0.7165`;
3. FI physical-unit bias is materially closer to zero than v2.4's
   approximately `+82 kW/m`;
4. FI/ROS MAE does not regress in exchange for CCC;
5. BP predictions remain unchanged within numerical tolerance during the
   behavior-only fine-tune;
6. v2.5 BP CCC, MAE, and top-10 IoU are preserved;
7. gains appear on held-out test hexels, not only validation;
8. the selected loss is confirmed across seeds 42, 1337, and 2024.

## Final intended sequence

```text
v2.4
  task-specific decoders + signed bounded BP hazard repair
    |
    v
v2.5
  BP-only redesign:
  separate attenuation, amplification, and missing-support hazard
  preferably on native 256x256 geometry
    |
    v
v2.6
  FI/ROS-only calibration:
  behavior-path fine-tuning with raw regression CCC
  no BP, propagation, decoder, or geometry redesign
```
