# Decision log

One line per decision that would be hard to reconstruct later. The dissertation's
methodology chapter is written from this file.

| Date | Decision | Why |
|---|---|---|
| 2026-10-14 | Scope confirmed at three research questions | Supervisor feedback: scope is fine, keep all three |
| 2026-10-16 | FP32 is the control; macro-F1 held within one point | Isolates calibration change from accuracy change (internal validity) |
| 2026-10-16 | Metrics implemented in-repo rather than imported | Binning scheme and "delivered precision" must be explicit and testable |

## 17 October 2026 — scope revision after supervisory feedback

Dr Minhas reviewed the outline and asked for four changes. All four are now
reflected in `Thesis-Outline_Ali-Hassan.docx` and in `configs/default.yaml`.

**The accuracy constraint in Q1a is no longer a filter.** Holding macro-F1
within one point would, as he pointed out, likely have excluded INT4 entirely —
the condition the thesis is named after. Macro-F1 is now recorded beside every
calibration measurement and treated as an observed covariate, so the question
becomes whether calibration degrades in proportion to accuracy or
disproportionately. `drop_cells_outside_band: false` enforces this in code.

**One corpus, two remedies.** AG News only. Threshold re-tuning is demoted from
a third treatment to the practitioner baseline, since it is what an engineer
would attempt first. Open Food Facts is recorded as a conditional extension,
attempted only if RQ1 and RQ2 complete early.

**AG News is balanced, which RQ1(b) cannot use.** RQ1(b) asks whether
calibration damage concentrates in low-frequency classes, and AG News has none.
Rather than reintroduce a second corpus, a controlled long-tailed subsample is
drawn from AG News itself at per-class keep rates of 1.0, 0.3, 0.1 and 0.03.
One download, one preprocessing step, and RQ1(b) survives the scope cut.

**Device measurement reports a ratio.** Absolute latency on a handset is
unstable under thermal throttling and background scheduling. The primary
quantity is now overhead relative to FP32 on the same device in the same
session, measured under a fixed protocol: standardised profiler, airplane mode,
discarded warm-up, interleaved conditions. A desktop CPU proxy is the declared
fallback.

## 17 October 2026 — positioning corrected after reading papers 35–37

Two earlier readings were wrong and the correction matters for RQ1 and RQ3.

**Xia et al. (2023)** does not pre-empt RQ1. It studies *convolutional vision*
models, and its finding is mechanistic: overconfidence partially explains why
accuracy is robust to quantisation, because low-confidence predictions are the
ones that flip. There is no text classification and no downstream abstention
decision. It must be cited, and the paper itself calls the dilemma
"underexplored".

**Shang et al. (CVPR 2024) and Edalati et al. (2024) are not prior art for
RQ3.** Both use "calibration" in the *quantisation* sense — selecting scaling
factors, zero points and Hessian-weighted updates — not in the
confidence-versus-correctness sense. The collision is terminological. This is
worth a sentence in the positioning section rather than a threat to the
contribution.

## 21 October 2026 — nearest prior work found; RQ2 reframed

A Google Scholar citation alert on Timbus and Lemnaru surfaced the only paper
citing them: **El Allam and Hamlich (2026)**, *Quantization-aware Mondrian
conformal prediction for embedded ECG classification*, Biomedical Signal
Processing and Control, DOI 10.1016/j.bspc.2026.111217. Free preprint at SSRN
5944060.

They ran the comparison at the centre of RQ2. Conformal calibration derived at
FP32 and applied to INT8 inference reached **87.84 per cent coverage against a
90 per cent target**; recalibrating with the deployed INT8 model restored it to
90.09 per cent. Pooled calibration concealed class-specific undercoverage that
Mondrian class-conditional calibration removed, cutting the class-coverage gap
from 3.06 to 0.12 percentage points. Profiled on an ESP32S3: 26.7 KB model,
conformal set construction under 0.1 ms.

**Consequence.** RQ2 can no longer ask whether a full-precision threshold still
holds after quantisation; that is now published. It is reframed as a question
of boundary and magnitude: how far the effect reaches in multi-class text at
INT4, measured through an abstention threshold at a precision target rather
than through prediction sets.

**What remains ours.** Text rather than 12-lead ECG; INT4 as well as INT8;
multi-class rather than binary; abstention at a precision target rather than
conformal set size; three confidence signals compared rather than conformal
alone; temperature re-fitting as a comparator, which they do not test; and
calibration error measured apart from accuracy, which is RQ1 and which their
coverage framing never addresses.

This is a stronger position than before. The premise is no longer speculative,
and the contribution becomes how far a demonstrated effect generalises.

## 23 October 2026 — data pipeline: splits, the long tail, and two senses of "calibration"

Ethics turned out not to be required for a secondary-data study. Dr Minhas
confirmed it, the dissertation team's own reminder states it, and written
confirmation has been requested from the team. Work therefore began earlier
than the roadmap assumed.

`scripts/fetch_data.py` pins AG News (`fancyzhx/ag_news`) and derives every
split. Each split is written as JSONL with a fixed field and row order before
being hashed, because only a canonical serialisation hashes the same way twice;
the download cache does not. `data/processed/manifest.json` records the counts
and digests and is committed, while the data itself is not.

**Three decisions worth defending in the methodology chapter.**

*The probability-calibration split is held out of training entirely.* 15 per
cent of AG News's training set, stratified, giving 18,000 rows. Temperature
scaling and split conformal prediction both need honest held-out data; reusing
training rows would flatter every remedy RQ3 evaluates, and the error would not
show up in the numbers.

*The PTQ calibration sample is a different thing that shares a name.* The 512
rows a quantisation backend uses to choose scaling factors are drawn from the
training portion, balanced across classes, because that is what a practitioner
would have at quantisation time. Keeping it apart from the probability
calibration split is the difference between a clean result and a leak.

*The long tail imbalances training only.* Keep rates of 1.0, 0.3, 0.1 and 0.03
give 25,500 / 7,650 / 2,550 / 765, an imbalance ratio of 33:1. RQ1(b) asks
whether calibration damage concentrates in classes the model saw rarely, so the
imbalance belongs in what the model learns from. The test set stays balanced at
1,900 per class, because measuring per-class calibration needs enough examples
of each class to measure.

**Leakage checked, not assumed.** `tests/test_data.py` asserts that the
calibration split shares no row with training, that the test split shares none
with either, that the long tail is a subset of training, and that the manifest
digests match the files on disk. A separate check found no exact duplicate
texts within any split and no text carrying conflicting labels. Near-duplicate
detection remains open and belongs in threats to validity, informed by papers
18 and 19.

## 23 October 2026 — TF-IDF baseline, simulated quantisation, pilot closed

`run_rq1.py` now runs end to end. Pipeline proven; the numbers below are a
pilot signal from the linear baseline, not a result.

**Quantisation is simulated for the linear model.** A TF-IDF classifier has no
ONNX graph worth quantising, so `src/qcal/quantise.py` maps values onto the
integer grid a backend would use and back again, so downstream arithmetic sees
the same rounding error. Weights are symmetric and per-output-class; the
activation range is observed on the PTQ calibration sample, never on test,
because a deployed model cannot see its test inputs. The transformer will use a
real backend; this keeps the baseline runnable on the laptop.

**Seed variance for this model is exactly zero.** Five seeds return identical
figures, because logistic regression fitted by lbfgs on fixed data is
deterministic. Reporting seed spread here would claim an uncertainty of zero
that no reader should believe, so intervals come from a percentile bootstrap
over the evaluation set instead (`bootstrap_ci`, 2000 resamples). Seeds are
retained because the transformer will genuinely vary across them.

**Two patterns worth testing properly.**

*Balanced split.* INT8 is inert: macro-F1 and ECE move within the bootstrap
interval. INT4 costs 0.51 points of macro-F1 while ECE rises from 0.0206 to
0.0257, a quarter again, with the intervals [0.0161, 0.0266] and [0.0214,
0.0322] barely separated. Disproportionate in direction but not yet
convincingly so in magnitude, which is the honest reading.

*Long-tailed split.* Two things happen at once. At FP32, per-class ECE runs
0.0182 for the majority class against **0.2139 for the rarest**, a twelvefold
gap: RQ1(b)'s premise holds plainly. And INT4 *improves* pooled ECE, 0.0469 to
0.0335, while macro-F1 is flat. The mechanism looks like Xia et al. (2023):
the model is overconfident, quantisation noise lowers mean confidence from
0.8131 to 0.7995, and the error shrinks by accident. The per-class figures show
what the pooled number hides, with World worsening 0.0182 to 0.0265 and Sports
0.0620 to 0.0762 while the rare class barely moves. This mirrors El Allam and
Hamlich's finding that pooled calibration conceals class-specific behaviour,
and it is an argument for reporting per-class throughout.

**Caveat on all of it.** One model family, simulated rather than backend
quantisation, no remedies applied, a single corpus. The transformer is the test
of whether any of this survives.

## 24 October 2026 — the calibration split was three times larger than it needed to be

The probability-calibration split was set at 15 per cent of training, 18,000
rows, chosen without asking what split conformal prediction actually requires.
It requires much less. The realised coverage of a split conformal predictor has
an irreducible spread of sqrt(alpha(1-alpha)/(n+2)): 0.39 percentage points at
6,000 calibration rows against 0.22 at 18,000. Three times the data buys 0.17
percentage points.

Against that it cost 12,000 rows of training data and tripled the inference
bill of every run, since probabilities are emitted for the calibration split as
well as the test split at all three precisions. A smoke test was still running
after forty minutes.

The fraction is now 5 per cent: 6,000 calibration rows, 114,000 for training.
Temperature scaling fits one parameter and needs far less again. All splits
were re-pinned and the manifest digests changed; nothing depended on the old
ones.

**The cross-machine interface is the probability matrix, not the model.** A
fine-tuned graph is 256 MB and the full grid is 7.5 GB, past GitHub's per-file
limit and with no business in a repository. `results/probs/*.npz` holds the
predicted probabilities for every precision on the test and calibration splits,
about 300 KB apiece, and every quantity in RQ1 and RQ2 is recoverable from
them. The graphs stay on the machine that made them and go to the handset
directly when RQ3 needs them.
