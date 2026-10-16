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
