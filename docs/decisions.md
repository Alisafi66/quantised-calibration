# Methods and design decisions

A technical reference for every choice the experiments depend on, with the
reason for each. The dissertation's methodology chapter is written from this
file. Where an approach was tried and rejected, section 7 records it, because
a reader reproducing this work needs to know which configurations do not work
and why.

---

## 1. Data

### Corpus

AG News (`fancyzhx/ag_news`), four balanced topic classes, 120,000 training
rows and 7,600 test rows. A single standard benchmark keeps the experimental
grid small enough that each cell can be analysed rather than merely run.

### Snapshot pinning

`scripts/fetch_data.py` writes every split as JSONL with a fixed field order
and a fixed row order, then records a SHA-256 digest in
`data/processed/manifest.json`. The manifest is committed; the data is not.

Hashing the written file rather than the download cache is deliberate. Only a
canonical serialisation produces the same digest twice. Files are opened with
`newline="\n"` because Python's text mode otherwise translates the terminator
to the host convention, so the same rows written on Windows and on macOS differ
by one byte per line and hash differently while being identical in content. A
digest that varies by operating system pins the machine rather than the corpus.

### Splits

| Split | Rows | Purpose |
|---|---|---|
| `train_balanced` | 114,000 | model fitting, 28,500 per class |
| `train_long_tailed` | 40,755 | model fitting under imbalance |
| `prob_calib` | 6,000 | temperature scaling and conformal prediction |
| `test` | 7,600 | evaluation, balanced at 1,900 per class |
| `ptq_calib` | 512 | quantisation scaling factors |

**The probability calibration split is held out of training entirely.**
Temperature scaling and split conformal prediction both require honest held-out
data. Reusing training rows would flatter every remedy under evaluation, and
the error would not be visible in the resulting numbers.

**Its size is 5 per cent of training, not more.** The realised coverage of a
split conformal predictor has an irreducible spread of
`sqrt(alpha(1-alpha)/(n+2))`: 0.39 percentage points at 6,000 calibration rows
against 0.22 at 18,000. Tripling the split buys 0.17 points of precision at a
cost of 12,000 training rows and three times the inference required to score
it. Temperature scaling fits a single parameter and needs far less again.

**The long tail imbalances training only**, at per-class keep rates of 1.0,
0.3, 0.1 and 0.03, giving 28,500 / 8,550 / 2,850 / 855 and a ratio of 33:1.
The question is whether behaviour degrades for classes the model saw rarely,
so the imbalance belongs in what the model learns from. The test set stays
balanced because measuring per-class quantities requires enough examples of
every class to measure.

### Leakage

Tested rather than assumed. `tests/test_data.py` asserts that the calibration
split shares no row with training, that the test split shares none with either,
that the long tail is a strict subset of training, and that every manifest
digest matches the file on disk. A separate check found no exact duplicate text
within any split and no text carrying conflicting labels. Near-duplicate
detection remains open and belongs in threats to validity.

---

## 2. Quantisation

### Two senses of the word

The term carries two unrelated meanings in this literature and the code keeps
them apart.

- **PTQ calibration** is the selection of scaling factors and zero points from
  a sample of representative inputs. `quantisation.calibration_samples` refers
  to this.
- **Probability calibration** is the agreement between a confidence score and
  the likelihood of correctness. `dataset.prob_calib_fraction` refers to this.

The PTQ sample is drawn from training data and balanced across classes, because
that is what a practitioner has available at quantisation time. It is disjoint
from the probability calibration split.

### INT8: static, restricted to MatMul

`quantise_int8` applies static quantisation with per-channel weights and an
activation range observed on the PTQ sample, restricted to MatMul operators.

The restriction is necessary, not cosmetic. Quantising every operator reduced
test accuracy from 0.89 to 0.27, and percentile clipping at 99.999 recovered it
only to 0.81 while driving ECE to 0.28. The cause is the activation-outlier
behaviour of transformer encoders described by Dettmers et al. (2022): a small
number of extreme features set the scale for an entire tensor and the remaining
values collapse into a few levels. LayerNorm and the attention softmax are left
in higher precision for the same reason a mobile deployment would leave them
there.

### INT4: weight-only, block-wise

`quantise_int4` uses ONNX Runtime's `MatMulNBitsQuantizer` at 4 bits with a
block size of 128. This runs on CPU and requires no GPU, so the CUDA machine is
needed for fine-tuning alone.

ONNX Runtime was chosen over `bitsandbytes` because ONNX Runtime Mobile is what
the handset will run. Quantising through the same runtime means the desktop
measurements and the device measurements describe one artefact rather than two
implementations that share a name.

Weight-only quantisation touches MatMul nodes and leaves the embedding table in
full precision, so the resulting graph is 116 MB against INT8's 133 MB and
FP32's 256 MB. Size claims must be stated with that qualification.

### The linear baseline

A TF-IDF classifier has no ONNX graph worth quantising, so `src/qcal/quantise.py`
simulates the arithmetic: values are mapped onto the integer grid a backend
would use and mapped back, so downstream computation sees the same rounding
error. Weights are symmetric and per-output-class; the activation range is
observed on the PTQ sample and never on test, because a deployed model cannot
see its test inputs.

This makes the baseline runnable without a GPU. It also means the linear and
transformer results are not directly comparable, since one is simulated and the
other passes through a real backend.

---

## 3. Metrics

Implemented in `src/qcal/metrics.py` rather than imported, so that every
definition reported in the dissertation is explicit and testable.

| Metric | Definition | Measures |
|---|---|---|
| ECE | equal-width binning, 15 bins (Guo et al., 2017) | calibration |
| Delivered precision | accuracy among answered inputs at a threshold | the kept promise |
| Coverage | fraction of inputs answered | usefulness |
| Threshold | most permissive value whose delivered precision still meets target | operating point |
| AURC | area under the risk-coverage curve | discrimination |
| Per-class ECE | ECE within each true class | where damage concentrates |

### The threshold definition

`threshold_for_target_precision` returns the **most permissive** threshold whose
delivered precision is still at or above target, which is what a practitioner
wants: answer as many inputs as possible while keeping the advertised precision.

It examines the whole ranking rather than scanning downward and stopping at the
first shortfall. Near the top of the ranking the answered set is a handful of
inputs, so a single mistake drops precision far below any target and a downward
scan halts on noise. In one case this returned a threshold answering 1.3 per
cent of inputs where another answered 80 per cent and kept the same promise.

The function returns `nan` when no threshold meets the target, which is a
finding about the model rather than an error.

### Why both ECE and AURC

Calibration and discrimination are orthogonal properties. ECE asks whether a
confidence of 0.9 is correct 90 per cent of the time. AURC asks whether more
confident predictions are in fact more likely correct. A model can be well
calibrated and poor at ranking, or the reverse. Selective prediction depends on
ranking, so reporting only ECE would miss the failure this study found.

---

## 4. Statistical procedure

### Paired within-seed comparison

Each quantised model is compared against the full-precision model trained from
the same seed on the same data. The quantity analysed is the per-seed
difference, tested with a two-sided t-test on 4 degrees of freedom, critical
value 2.776. The consistency of the sign across seeds is reported alongside,
because with five observations a consistent direction is as informative as the
statistic.

### Why a bootstrap over the test set is not sufficient on its own

A bootstrap that resamples evaluation rows measures how far a figure would move
on a different sample of the same corpus. It says nothing about how far it
would move had the model been trained again.

This distinction is not academic. On one seed, INT4 appeared to raise ECE by
+0.0056 with a paired bootstrap interval excluding zero and the quantised model
worse in all 4,000 resamples. Across five seeds the same comparison gives a mean
of +0.0006 with mixed signs and t(4) = +0.39. The seed-to-seed standard
deviation of FP32 ECE alone is 0.0109, roughly twice the apparent effect.
Training variability dominates, and a bootstrap cannot see it.

The bootstrap is retained for reporting uncertainty on a single configuration,
for which it is the right instrument, and for the deterministic linear
baseline, where five seeds return identical figures and seed spread would claim
an uncertainty of zero.

---

## 5. Results

DistilBERT, two epochs, five seeds, three precisions, two variants. Paired
within-seed change from FP32.

| | balanced INT8 | balanced INT4 | long-tailed INT8 | long-tailed INT4 |
|---|---|---|---|---|
| macro-F1 | **-0.0019** t=-5.44 | -0.0000 | **-0.0242** t=-4.53 | -0.0013 |
| ECE | -0.0027 t=-1.43 | -0.0002 | -0.0029 t=-0.68 | +0.0006 |
| coverage at 95% | **-0.33pp** t=-3.13 | -0.06pp | **-8.12pp** t=-8.34 | -0.74pp |
| coverage at 98% | **-2.11pp** t=-3.69 | -0.09pp | **-11.19pp** t=-13.02 | -0.64pp |

Bold marks significance at 5 per cent with a consistent sign across all five
seeds.

### What holds across every cell

**ECE detects nothing.** Every ECE t-statistic is below 1.5 with mixed signs,
in both variants at both precisions.

**Coverage at a fixed precision target detects INT8 every time**, with the same
sign in all five seeds, and the effect grows as the promise tightens.

**Imbalance amplifies but does not create.** The effect is roughly
twenty-five times larger on the long-tailed split, and it remains significant
on the balanced one.

**INT4 is benign throughout.** Weight-only block-wise INT4 preserved selective
behaviour in every condition tested while static INT8 did not. The quantisation
scheme determines the risk, not the bit width.

### The mechanism

Four analyses establish what is breaking.

**The threshold keeps its precision promise.** Promised minus delivered at a
95 per cent target is +0.28 points at long-tailed INT8 against +0.21 at FP32.
The shortfall is calibration-to-test generalisation and quantisation barely
adds to it. The model keeps the promise and pays in coverage.

**Re-tuning does not recover the coverage.** A fresh threshold fitted on the
quantised model's own calibration split leaves long-tailed INT8 at -10.26
points, t(4) = -3.48, marginally worse than not re-tuning. The threshold was
never misplaced.

**No threshold can.** An oracle threshold fitted on the test set itself still
loses 9.71 points at long-tailed INT8 and 0.42 on the balanced split.

**Because the loss is in the ranking.** AURC rises by 0.01009 at long-tailed
INT8, t(4) = +3.83. The model orders its own correctness less well. ECE is
indifferent to ordering, which is why it reports nothing.

**Temperature scaling confirms this.** A single temperature rescales every
logit vector by the same scalar. It can shift the ranking slightly, since
examples whose logits are spread differently respond differently, but it
cannot restructure it. On long-tailed INT8 it improves ECE markedly, -0.0243
with t = -3.21, moves AURC by -0.0008 against the +0.0101 that quantisation
introduced, and leaves the coverage gap at -9.13 points. It repairs the
calibration and does not repair the failure.

This predicts the behaviour of split conformal prediction: it should restore
the validity of the guarantee, since its coverage is distribution-free by
construction, while failing to restore efficiency, because it consumes the same
damaged ordering. Valid coverage purchased with larger prediction sets.

### Confidence signals

Softmax, margin and entropy lose 8.12, 8.06 and 8.22 points of coverage
respectively at long-tailed INT8. They are three readings of the same ordering,
so the choice between them does not affect the result.

### Class frequency

Per-class ECE at FP32 averages 0.0085 for the class with 28,500 training
examples against 0.1358 for the class with 855, a factor of sixteen. This is an
effect of imbalance and is present before any compression is applied.

---

## 6. Environment

Fine-tuning is the only stage requiring a GPU. Quantisation, evaluation and
analysis run on CPU through ONNX Runtime.

**The artefact that crosses between machines is the probability matrix, not the
model.** A fine-tuned graph is 256 MB and the full grid is 7.5 GB, beyond
GitHub's per-file limit. `results/probs/*.npz` holds the predicted
probabilities for every precision on the test and calibration splits at roughly
300 KB each, and every quantity in RQ1 and RQ2 is recoverable from them. The
graphs remain on the machine that produced them and go to the handset directly
when device measurement requires them.

Dependencies are limited to what the code imports. An earlier list carried
packages that were never used, and their version pins sent pip's resolver
through 200,000 rounds of backtracking on a clean install before it failed.

---

## 7. Approaches tried and rejected

| Approach | Outcome | Reason |
|---|---|---|
| Static INT8 over all operators | accuracy 0.89 to 0.27 | activation outliers set tensor scales |
| Static INT8, percentile clipping at 99.999 | accuracy 0.81, ECE 0.28 | clipping discards the outliers that carry signal |
| Dynamic INT8 | shape inference error on the exported graph | incompatible with the dynamo exporter's output |
| Calibration split at 15 per cent | withdrawn | 0.17 points of conformal precision for 12,000 training rows |
| Downward threshold scan halting at first shortfall | withdrawn | halts on noise at the top of the ranking |
| Inference batch of 256 | out of memory | the GELU intermediate is batch x 128 x 3072 floats |
| `bitsandbytes` for INT4 | not required | ONNX Runtime does weight-only INT4 on CPU |

---

## 8. Open items

- RQ1(c), the comparison between the linear baseline and the transformer, is
  not yet answerable: one uses simulated quantisation and the other a real
  backend.
- Split conformal prediction is implemented in neither the metrics module nor
  the experiment scripts.
- Device measurement has not begun and the handset model is unspecified.
- Near-duplicate detection within AG News remains open.
