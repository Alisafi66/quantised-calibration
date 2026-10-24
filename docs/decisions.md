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

## 24 October 2026 — the threshold definition was wrong, and untested

`threshold_for_target_precision` scanned confidences downward and stopped at
the first point where delivered precision fell below target. Near the top of
the ranking the answered set is a handful of inputs, so one mistake drops
precision far below any target and the scan halted on noise. On the pilot it
returned 0.9819, answering 1.3 per cent of inputs, where 0.8631 answers 80 per
cent and keeps the same promise.

It had no test. Three now cover it: that it takes the largest qualifying
answered set, that a single error at the top does not derail it, and that it
returns nan when no threshold can keep the promise, which is itself a finding
rather than an error.

The definition now reads: the most permissive threshold whose delivered
precision is still at or above target. That is what a practitioner wants,
answer as much as possible while keeping the advertised precision, and it is
the quantity RQ2 carries across precisions.

**A second failure mode for RQ2.** On the pilot, a threshold fitted at FP32 and
carried unchanged held its precision at INT8 but coverage collapsed from 79 to
50 per cent: quantisation suppressed confidence, so the model abstained far
more rather than answering wrongly. El Allam and Hamlich report the opposite
direction, undercoverage of the promise. Both are failures of transfer, and RQ2
should expect either. Worth stating in the thesis that the threshold can fail
by becoming too cautious as well as too permissive.

## 24 October 2026 — first real transformer results: the effect is conditional

DistilBERT, 2 epochs, AG News, seed 0, on the CUDA machine. Probabilities
crossed back as npz; every figure below was recomputed on the laptop.

**Balanced split: nothing happens.** Macro-F1 0.9452 at FP32 against 0.9446 at
INT8 and 0.9461 at INT4; ECE 0.0092, 0.0100, 0.0109. A paired bootstrap on the
difference, resampling the same test rows for both models, puts dECE at
+0.0009 [-0.0028, +0.0035] for INT8 and +0.0018 [-0.0015, +0.0043] for INT4.
Both intervals contain zero. A threshold fitted at FP32 and carried across
tracks to the fourth decimal at every target from 0.95 to 0.995. On a corpus
this easy, with a model this well calibrated, there is nothing to damage.

**Long-tailed split: the effect appears, and it is specific.**

INT4 degrades calibration significantly while accuracy barely moves. ECE rises
from 0.0460 to 0.0516, a paired difference of +0.0056 [+0.0024, +0.0091] with
the quantised model worse in every one of 4,000 resamples, while macro-F1 falls
only 0.0071. That is the thesis claim, measured: calibration damage out of
proportion to accuracy loss.

INT8 does not damage ECE, +0.0008 [-0.0050, +0.0041], but for an instructive
reason. It loses more accuracy than INT4 (macro-F1 0.8421) *and* loses
confidence (mean 0.8848 against 0.9053), and the two roughly cancel in a pooled
calibration figure. Pooled ECE conceals it; the selective-prediction view does
not.

**RQ1(b) is answered.** Per-class ECE at FP32 runs 0.0039 for the class with
28,500 training examples and 0.1148 for the class with 855, a factor of
twenty-nine before any quantisation. Quantisation then adds to the rare class
specifically: Sci/Tech reaches 0.1486 at INT8 and 0.1318 at INT4. The damage
concentrates where the model saw least.

**RQ2 is where INT8's loss surfaces.** A threshold fitted at FP32 for a 95 per
cent target answers 60.5 per cent of inputs at FP32 and only 54.4 per cent at
INT8, a loss of 6.2 points of coverage, while also delivering lower precision,
0.9395 against 0.9472. At a 97 per cent target the coverage loss is 8.8 points,
24 per cent in relative terms. At 99 per cent INT8 answers 9 inputs in 10,000
against FP32's 647. The carried threshold fails in both directions at once: it
delivers less precision than promised and abstains far more than intended.

**A baseline that must be reported.** FP32 itself misses the target at every
level, delivering 0.9472 against a 0.95 promise. That is the calibration-to-test
generalisation gap, present without any quantisation. The quantisation effect
sits on top of it, and a study that failed to report the FP32 transfer gap
would attribute ordinary generalisation error to compression.

## 24 October 2026 — the snapshot hashed differently on Windows

Running `fetch_data.py` on the CUDA machine produced a manifest whose digests
disagreed with the laptop's for every split, including the raw snapshot, while
every row count matched exactly. The cause was `write_jsonl` opening files in
text mode: Python translates `\n` to the platform's convention, so Windows
wrote CRLF and macOS wrote LF, one byte per line of difference across 114,000
rows.

The content was never in question and no result is affected: the same rows,
the same labels, the same order, and the probabilities already produced remain
valid. What was wrong was the claim the hash makes. A digest that changes with
the operating system pins the machine rather than the corpus, which is the
opposite of what a reproducibility record is for.

`newline="\n"` is now passed explicitly and a test asserts no carriage returns
reach the file. Both machines must re-run `fetch_data.py` once so their
manifests agree.

## 24 October 2026 — five seeds overturn the INT4 claim and establish an INT8 one

Seed 0 showed INT4 raising ECE by +0.0056 with a paired bootstrap interval
excluding zero and the quantised model worse in all 4,000 resamples. Across
five seeds that effect disappears: per-seed dECE of +0.0056, -0.0019, +0.0009,
+0.0016, -0.0032, mean +0.0006, t(4) = +0.39, signs mixed. **Seed 0 was an
outlier and the earlier claim was wrong.**

The error was methodological and worth recording. The bootstrap resamples test
rows, so it measures how far a figure would move on a different sample of the
same corpus. It says nothing about how far it would move had the model been
trained again. Here training variability dominates: the seed-to-seed standard
deviation of FP32 ECE alone is 0.0109, twice the effect seed 0 appeared to
show. A tight bootstrap interval on one seed is not evidence about quantisation;
it is evidence about that one model.

**What does survive, and it is specific to INT8.**

Macro-F1 falls by 0.0242 at INT8, t(4) = -4.53, negative in every seed. INT4
costs 0.0013, t = -0.77, mixed.

Coverage at a 95 per cent precision target, with the threshold fitted at FP32
and carried across, falls by **8.12 percentage points at INT8**, sd 2.18,
t(4) = -8.34, negative in all five seeds and ranging from -5.5 to -10.4. INT4
loses 0.74 points and is not consistent.

**Pooled ECE does not detect any of it.** The INT8 dECE is -0.0029 with
t(4) = -0.68: by that instrument INT8 looks harmless, or even slightly
improved, while it is discarding eight per cent of the inputs the full-precision
model would have answered. INT8 loses accuracy and loses confidence at the same
time, and in an averaged calibration figure the two cancel.

That is the finding. Not that quantisation damages calibration as ECE measures
it, but that **it damages selective-prediction behaviour in a way ECE cannot
see**, which is precisely the case for reporting coverage at a fixed precision
alongside calibration error rather than instead of it. RQ2 was the right
question; RQ1's instrument is the one that comes up short.

**RQ1(b) survives independently.** Per-class ECE at FP32 averages 0.0085 for the
class with 28,500 training examples against 0.1358 for the class with 855, a
factor of sixteen across seeds. That is an effect of imbalance rather than of
quantisation, and it holds before any compression is applied.

## 24 October 2026 — the complete grid, and a result that does not depend on the corpus

Ten runs: two variants, five seeds, DistilBERT, three precisions. Paired
within-seed changes from FP32, two-sided t on 4 df, critical value 2.776.

| | balanced | | long-tailed | |
|---|---|---|---|---|
| | INT8 | INT4 | INT8 | INT4 |
| macro-F1 | **-0.0019** t=-5.44 | -0.0000 | **-0.0242** t=-4.53 | -0.0013 |
| ECE | -0.0027 t=-1.43 | -0.0002 | -0.0029 t=-0.68 | +0.0006 |
| coverage @95% | **-0.33pp** t=-3.13 | -0.06pp | **-8.12pp** t=-8.34 | -0.74pp |
| coverage @98% | **-2.11pp** t=-3.69 | -0.09pp | **-11.19pp** t=-13.02 | -0.64pp |

Three things hold across every cell.

**Expected calibration error never detects the effect.** Not on the balanced
split, not on the long-tailed one, at neither precision. Every ECE t-statistic
is below 1.5 and the signs are mixed. A study measuring only ECE would conclude
that quantisation is harmless here.

**Coverage at a fixed precision target always detects INT8.** Significant in
both variants at both targets, with the same sign in all five seeds every
time, and the effect grows as the promise tightens: on the long-tailed split
2.1 points of coverage at a 95 per cent target become 11.2 at 98 per cent.

**The effect is not an artefact of imbalance.** It is twenty-five times larger
on the long-tailed split, but it is present and significant on the balanced one
too. Imbalance amplifies it; it does not create it.

**INT4 is benign throughout**, which is worth stating plainly because it
inverts the usual expectation. Weight-only block-wise INT4 preserved selective
behaviour in every condition tested, while static INT8 did not. For a
practitioner the actionable form is: the bit width is not what determines the
risk, the quantisation scheme is.

The thesis claim is therefore narrower and better evidenced than the one the
outline set out with. Not that quantisation degrades confidence calibration,
which this data does not support, but that **it degrades selective-prediction
behaviour in a way the standard calibration metric cannot see**, and that the
degradation tracks the quantisation scheme rather than the bit width.

## 24 October 2026 — the mechanism: ranking, not calibration

Four further analyses on the completed grid settle what is actually breaking.

**The threshold keeps its precision promise.** Promised minus delivered at a
95 per cent target is +0.19 pp at INT8 on the balanced split and +0.28 pp on
the long-tailed one, against +0.16 and +0.21 at FP32. The shortfall is a
calibration-to-test generalisation gap, and quantisation barely adds to it.
Q2(a)'s premise, that the quantised model would deliver *less precision* than
promised, is not what happens. It keeps the promise and pays in coverage.

**Re-tuning the threshold does not recover the coverage.** Fitting a fresh
threshold on the quantised model's own calibration split leaves long-tailed
INT8 at -10.26 pp against FP32, t(4) = -3.48, slightly worse than the -8.12 pp
without re-tuning. The threshold was never misplaced.

**Nor can any threshold.** An oracle threshold, fitted on the test set itself,
still loses 9.71 pp at long-tailed INT8 (t = -3.16) and 0.42 pp on the
balanced split (t = -8.46). No choice of operating point recovers what was
lost.

**Because what was lost is ranking, not calibration.** AURC rises by 0.01009
at long-tailed INT8, t(4) = +3.83: the model has become worse at ordering its
own correctness. ECE is indifferent to ordering, which is precisely why it
reports nothing. Calibration and discrimination are orthogonal, and selective
prediction depends on the one quantisation damaged.

**Temperature re-fitting confirms it, and answers Q3(a) ahead of schedule.**
A single temperature is a monotone rescaling and cannot reorder anything. On
long-tailed INT8 it improves ECE markedly, -0.0243 (t = -3.21), leaves AURC
untouched, -0.0008 (t = -1.92), and leaves the coverage gap at -9.13 pp
(t = -3.67) against -8.12 before. **It fixes the calibration and does not fix
the problem.** The remedy the literature prescribes for miscalibration cannot
help, because miscalibration is not what occurred.

This yields a prediction for Q3(b): split conformal will restore the
*validity* of the guarantee, since its coverage is distribution-free by
construction, but it cannot restore efficiency, because it too consumes the
damaged score ordering. Expect valid coverage purchased with larger prediction
sets.

**Q2(c) answered: no confidence signal is more robust.** Softmax, margin and
entropy lose -8.12, -8.06 and -8.22 pp respectively at long-tailed INT8. They
are three readings of the same damaged ordering, so the choice between them
does not matter.
