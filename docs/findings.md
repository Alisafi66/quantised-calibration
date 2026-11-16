# Findings

> **Static activation calibration destroys selective prediction. Expected
> calibration error cannot detect it, and dynamic quantisation at the same bit
> width avoids it.**

Every figure below is coverage at a **95 per cent delivered-precision
threshold**, fitted once on the full-precision calibration split and carried
across unchanged, which is the position of a practitioner who sets a threshold
before quantising and does not revisit it. Paired within-seed tests on five
seeds, 4 degrees of freedom, critical t of 2.776. The split is named on every
row because the effect size depends on it.

## Overview

| | Claim | Evidence | Split |
|---|---|---|---|
| **F1** | Static INT8 removes 8.12 points of coverage | t(4) = -8.34, 5/5 seeds | long-tailed |
| **F2** | The fixed calibration causes 82 per cent of it | +6.64 pp recovered, t = +5.97 | long-tailed |
| **F3** | Eight-bit weight-only quantisation is free | -0.01 pp, t = -0.40 | long-tailed |
| **F4** | INT4's benignity was weight-only, not bit width | 4 vs 8 bits: -0.72 pp, t = -2.76 | long-tailed |
| **F5** | Imbalance amplifies the effect, it does not create it | -6.04 pp per decade, R2 = 0.966 | five ratios |
| **F6** | The effect is not an artefact of the starved classes | -5.78 pp with the two rarest removed | long-tailed |
| **F7** | Expected calibration error is blind in every condition | all \|t\| below 0.70, six arms | long-tailed |

---

## F1. Static INT8 removes 8.12 points of coverage at a 95 per cent precision target

`SELECTIVE PREDICTION` `INT8 STATIC` `95% TARGET` `LONG-TAILED` `5 SEEDS`

| Precision | Coverage | Against FP32 | t(4) | Signs |
|---|---|---|---|---|
| FP32 | 67.36% | | | |
| INT8 static | 59.24% | **-8.12 pp** | -8.34 | 5/5 |
| INT4 weight-only | 66.62% | -0.74 pp | -2.57 | 5/5 |

Eight points of inputs that were answered at full precision fall below the
carried threshold after quantisation, so the deployed system silently answers
fewer cases at the precision it was configured for.

---

## F2. The cause is the fixed activation calibration, not the bit width

`MECHANISM` `STATIC VS DYNAMIC` `95% TARGET` `LONG-TAILED` `5 SEEDS`

Six arms, each differing from its neighbour in one property, holding the set
of quantised operators constant except where coverage is the property tested.

| Arm | Bits | Activations | Weights | Matmuls | Coverage | vs FP32 | t(4) |
|---|---|---|---|---|---|---|---|
| FP32 | 32 | float | float | 0 | 67.36% | | |
| `int8_static` | 8 | static, calibrated | per-channel | 48 | 59.24% | -8.12 | -8.34 |
| `int8_static_weights` | 8 | static, calibrated | per-channel | 36 | 59.47% | -7.89 | -8.19 |
| `int8_dynamic` | 8 | dynamic, per input | per-channel | 36 | 66.12% | -1.24 | -1.54 |
| `int8_weight_only` | 8 | float | block-128 | 36 | 67.35% | -0.01 | -0.40 |
| `int4_block128` | 4 | float | block-128 | 36 | 66.62% | -0.74 | -2.57 |
| `int4_block32` | 4 | float | block-32 | 36 | 66.97% | -0.39 | -1.65 |

Decomposing the 8.12 points, one property at a time:

| Property | Contrast | Effect | t(4) | Share |
|---|---|---|---|---|
| Attention matmuls, 48 against 36 | `int8_static_weights` / `int8_static` | +0.23 pp | +2.79 | 2.8% |
| **Activation scale source** | `int8_dynamic` / `int8_static_weights` | **+6.64 pp** | **+5.97** | **81.8%** |
| Activation quantisation at all | `int8_weight_only` / `int8_dynamic` | +1.23 pp | +1.52 | 15.1% |
| Bit width, 4 against 8 | `int4_block128` / `int8_weight_only` | -0.72 pp | -2.76 | |
| Weight granularity, 32 against 128 | `int4_block32` / `int4_block128` | +0.35 pp | +1.71 | |

Recomputing activation scales per input instead of fixing them from a 512-row
PTQ sample returns 6.64 of the 8.12 points and leaves a residual that is not
distinguishable from zero, so the damage is done by the calibration step
rather than by eight-bit arithmetic.

The attention contrast clears the critical t marginally at 0.23 points of
coverage. It is reported as negligible in size rather than as null.

---

## F3. Eight-bit weight-only quantisation costs nothing measurable

`INT8 WEIGHT-ONLY` `95% TARGET` `LONG-TAILED` `5 SEEDS`

| Arm | Coverage | Against FP32 | t(4) |
|---|---|---|---|
| FP32 | 67.36% | | |
| `int8_weight_only` | 67.35% | -0.01 pp | -0.40 |

Quantising the 36 weight projections to eight bits while leaving activations in
float reproduces full-precision selective behaviour, which rules out the
weights, the bit width and the block structure as sources of the loss in F1.

---

## F4. INT4's apparent benignity follows from its being weight-only

`INT4` `BIT WIDTH` `95% TARGET` `LONG-TAILED` `5 SEEDS`

| Contrast | Difference | t(4) | Signs |
|---|---|---|---|
| `int4_block128` against `int8_weight_only` | -0.72 pp | -2.76 | 4/5 |
| `int4_block32` against `int4_block128` | +0.35 pp | +1.71 | 4/5 |

Holding the quantised operator set and the absence of activation quantisation
constant, four bits against eight does not reach significance. The published
INT4 arm is benign because it never performs a calibration step, not because
four-bit weights are safer than eight-bit ones.

---

## F5. Imbalance amplifies the effect and does not create it

`DOSE RESPONSE` `IMBALANCE` `95% TARGET` `FIVE RATIOS` `5 SEEDS EACH`

| Ratio | Split | FP32 | INT8 change | t(4) | Signs | Loss as share of FP32 |
|---|---|---|---|---|---|---|
| 1:1 | `balanced` | 99.24% | -0.33 pp | -3.13 | 5/5 | 0.3% |
| 3:1 | `lt_r3` | 96.12% | -1.05 pp | -5.17 | 5/5 | 1.1% |
| 10:1 | `lt_r10` | 84.69% | -4.67 pp | -5.21 | 5/5 | 5.5% |
| 33:1 | `long_tailed` | 67.36% | -8.12 pp | -8.34 | 5/5 | 12.1% |
| 100:1 | `lt_r100` | 26.70% | -12.00 pp | -5.97 | 5/5 | 44.9% |

Log-linear fit: **-6.04 points of coverage per decade of imbalance, R2 = 0.966,
p = 0.0027**, Spearman rho = -1.000.

The effect is present and significant on a perfectly balanced corpus, so
imbalance sets the magnitude rather than producing the phenomenon. The final
column rules out a ceiling effect: the loss grows faster in relative terms than
in absolute ones, which a ceiling could not do. At 100:1 full precision itself
answers only 26.70 per cent of inputs, so that ratio establishes the direction
of the curve and is not an operating point.

---

## F6. The loss is not confined to the classes the long tail starved

`DATA SENSITIVITY` `CLASS STRUCTURE` `95% TARGET` `LONG-TAILED` `5 SEEDS`

| Test set | FP32 | INT8 | Change | t(4) | Signs |
|---|---|---|---|---|---|
| All four classes | 67.36% | 59.24% | -8.12 pp | -8.34 | 5/5 |
| Rarest class removed | 81.74% | 73.60% | -8.14 pp | -5.28 | 5/5 |
| Two rarest removed | 93.81% | 88.02% | -5.78 pp | -3.93 | 5/5 |

| Class | Training rows | FP32 | INT8 | Change | t(4) |
|---|---|---|---|---|---|
| World | 28,500 | 95.55% | 94.48% | -1.06 pp | -3.46 |
| Sports | 8,550 | 92.06% | 81.56% | -10.51 pp | -3.47 |
| Business | 2,850 | 57.60% | 44.75% | -12.85 pp | -6.22 |
| Sci/Tech | 855 | 24.23% | 16.17% | -8.06 pp | -2.39 |

Removing the starved classes leaves the effect intact, the worst-hit class is
not the rarest, and World, which was never subsampled, still loses coverage.
The damage does not sit where the subsampling was applied.

---

## F7. Expected calibration error detects none of it

`ECE` `INSTRUMENT` `15 BINS` `LONG-TAILED` `5 SEEDS`

| Arm | ECE | Against FP32 | t(4) | Detected |
|---|---|---|---|---|
| FP32 | 0.0485 | | | |
| `int8_static` | 0.0457 | -0.0029 | -0.68 | no |
| `int8_static_weights` | 0.0467 | -0.0018 | -0.44 | no |
| `int8_dynamic` | 0.0489 | +0.0004 | +0.22 | no |
| `int8_weight_only` | 0.0485 | -0.0000 | -0.12 | no |
| `int4_block128` | 0.0491 | +0.0006 | +0.39 | no |
| `int4_block32` | 0.0478 | -0.0007 | -0.69 | no |

Across six quantisation arms, including one that costs 7.89 points of coverage,
no ECE difference approaches significance and the signs are mixed. A metric
that averages confidence against accuracy over bins is insensitive to a change
in the ordering of confidences near a decision threshold, which is what
selective prediction depends on.

---

## Scope

One corpus (AG News, pinned 23 October 2026), one architecture (DistilBERT),
one runtime (ONNX Runtime, CPU). The imbalance axis is measured at five ratios
and the scheme axis at six arms, both on the long-tailed split unless stated.
Generalisation across corpora and architectures is untested.

## Reproduction

| Finding | Script | Output |
|---|---|---|
| F1, F5, F6 | `experiments/run_data_sensitivity.py` | `results/raw/data_sensitivity.json` |
| F2, F3, F4, F7 | `experiments/run_scheme_analysis.py` | `results/raw/scheme_decomposition_long_tailed.json` |

Both read committed probability matrices in `results/probs/` and need no GPU.
The matrices are produced by `experiments/train_transformer.py` and
`experiments/run_scheme_decomposition.py` on the CUDA machine.
