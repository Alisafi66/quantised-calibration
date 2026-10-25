# Selective prediction under post-training quantisation

Research code for an MSc dissertation (M599, 25 ECTS) at Gisma University of
Applied Sciences.

**Question.** When a classifier is quantised to run on a device, its accuracy is
checked but its confidence is not. If the model is allowed to abstain, that
decision rests on a threshold calibrated at full precision. Does that threshold
still deliver what it promised after compression, and if not, can it be repaired
within the device's latency and memory budget?

## Research questions

| | |
|---|---|
| **RQ1** | How far does post-training quantisation distort confidence estimates, independently of accuracy loss? |
| **RQ2** | How far does an abstention threshold tuned at full precision fail after quantisation, in multi-class text and at INT4? |
| **RQ3** | Can the guarantee be restored within on-device latency and memory budgets? |

## Status

| Stage | State |
|---|---|
| Data pipeline, pinned and leakage-tested | complete |
| TF-IDF baseline with simulated quantisation | complete |
| Transformer, ONNX INT8 and INT4, 10 runs | complete |
| RQ1 and RQ2 analysis | complete |
| Temperature re-fitting | complete |
| Split conformal prediction | not started |
| Device measurement | not started |

Headline result and the full statistical treatment are in
[`docs/decisions.md`](docs/decisions.md).

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_data.py --config configs/default.yaml
python -m pytest -q
```

`fetch_data.py` downloads AG News, derives every split, and records a SHA-256
digest for each in `data/processed/manifest.json`. The tests verify those
digests and assert that no split leaks into another.

### The linear baseline

Runs on CPU in about a minute per variant.

```bash
python experiments/run_rq1.py --variant balanced
python experiments/run_rq1.py --variant long_tailed
```

### The transformer

Fine-tuning wants a GPU. See [`docs/cuda-box-setup.md`](docs/cuda-box-setup.md)
for the CUDA machine, including a `--smoke` run that exercises the whole chain
in a few minutes before committing to a long job.

```bash
python experiments/train_transformer.py --variant balanced    --seeds 0 1 2 3 4
python experiments/train_transformer.py --variant long_tailed --seeds 0 1 2 3 4
```

This fine-tunes, exports to ONNX, quantises to INT8 and INT4, and writes
predicted probabilities to `results/probs/`. Those probability files are what
moves between machines; the 256 MB graphs stay where they were produced.

## Layout

```
src/qcal/        data loading and splits, models, quantisation, metrics
experiments/     one script per research question, plus transformer training
configs/         the full specification of a run: seeds, precisions, targets
results/probs/   predicted probabilities, the cross-machine artefact
results/         figures, tables, raw measurement output
tests/           unit tests for the metrics and guards against data leakage
docs/            methods record, environment setup, supervision log
```

## Notes on reading the code

- `configs/default.yaml` specifies a run completely. Nothing is hard-coded.
- **Macro-F1 is an observed covariate, not a filter.** A precision whose
  accuracy falls further is interpreted rather than excluded.
  `drop_cells_outside_band: false` enforces this.
- **Calibration means two different things.**
  `quantisation.calibration_samples` is the PTQ sample used to choose scaling
  factors. `dataset.prob_calib_fraction` is the held-out split used for
  temperature scaling and conformal prediction. The two are disjoint.
- Metrics are implemented in `src/qcal/metrics.py` rather than imported, so
  that every definition reported in the dissertation is explicit and tested.
- Comparisons are paired within seed and tested across seeds. A bootstrap over
  the evaluation set alone understates uncertainty, for reasons set out in
  `docs/decisions.md`.

## Author

Abdelkerim Ali Hassan, GH1046256
MSc Data Science, Artificial Intelligence and Digital Business
Supervisor: Dr. Nasir Mehmood Minhas
