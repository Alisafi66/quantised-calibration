# Optimising Confidence Calibration in Quantised On-Device Classifiers

Research code for an MSc dissertation (M599, 25 ECTS) at Gisma University of
Applied Sciences.

**Question.** When a classifier is quantised to run on a device, its accuracy is
checked but its *confidence* is not. If the model is allowed to abstain, that
decision rests on a threshold calibrated at full precision. Does that threshold
still deliver the precision it promised after compression — and if not, can it be
repaired within the device's latency and memory budget?

## Research questions

| | |
|---|---|
| **RQ1** | How far does post-training quantisation distort confidence estimates, independently of accuracy loss? |
| **RQ2** | Does an abstention threshold tuned at full precision still deliver its promised precision at INT8 and INT4? |
| **RQ3** | Can the guarantee be restored within on-device latency and memory budgets? |

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_data.py   --config configs/default.yaml   # pins + hashes the snapshot
python experiments/run_rq1.py  --config configs/default.yaml   # calibration across precisions
python experiments/run_rq2.py  --config configs/default.yaml   # promised vs delivered precision
python experiments/run_rq3.py  --config configs/default.yaml   # remedies + device cost
python scripts/make_figures.py                                 # regenerates every figure
```

Every figure and table in the dissertation is produced by these commands from the
pinned snapshots. No result is hand-edited.

## Layout

```
src/qcal/        library code: data, models, quantisation, calibration, metrics
experiments/     one script per research question
configs/         the full specification of a run — seeds, precisions, tolerances
results/         figures, tables, raw measurement output
tests/           unit tests for the metric implementations
docs/            notes, logs, meeting records
```

## Method notes

- **Control condition** is FP32. Macro-F1 is held within one point across
  precisions so calibration moves while accuracy does not.
- **Five seeds** per configuration, confidence intervals on every figure,
  bootstrap on the headline comparison.
- **Baselines** are the majority class and the full-precision model, reported
  beside every result.
- *Calibration* in `quantisation.calibration_samples` means the PTQ calibration
  **set**, not probability calibration. The two senses are kept separate
  throughout the code.

## Author

Abdelkerim Ali Hassan · GH1046256 · MSc Data Science, AI and Digital Business
Supervisor: Dr. Nasir Mehmood Minhas
