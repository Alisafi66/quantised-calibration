# Round two: the imbalance sweep

The reported coverage loss rests on one long tail, drawn at keep rates
1.0 / 0.3 / 0.1 / 0.03. That is a parameter chosen by the author, and a result
that holds only at the parameter chosen is an artefact. This round measures the
same effect at five imbalance ratios so that the answer is a dose-response
curve rather than a single point.

Nothing about the corpus changes. AG News stays pinned to the snapshot of
23 October 2026, the test split and the probability-calibration split are not
touched, and the long tails are drawn from the same `train_balanced` pool that
produced the published result. What changes is how severely that pool is
subsampled before fine-tuning.

Because the training distribution changes, the models must be fine-tuned
again. This is not a re-analysis of existing predictions.

---

## Before leaving the laptop

Push the two new scripts and the trainer change, or the workstation will pull a
tree that cannot run the sweep.

```bash
cd ~/Developer/quantised-calibration
git add experiments/run_data_sensitivity.py scripts/make_imbalance_sweep.py experiments/train_transformer.py docs/round-two-runbook.md
git commit -m "Add the imbalance sweep and the data-sensitivity analysis"
git push
```

---

## On the workstation, from a cold boot

PowerShell, from the repository root. If the machine has been off since the
last run, the virtual environment is intact but not active.

```powershell
cd $HOME\quantised-calibration
git pull
.venv\Scripts\activate
```

Confirm the GPU is visible before committing hours to it.

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

This must print `True NVIDIA GeForce RTX 4080 SUPER`. If it prints `False`, the
CPU build of torch is installed; reinstall from the CUDA index as in
`cuda-box-setup.md` and do not proceed until it prints `True`.

### Derive the new splits

```powershell
python scripts\make_imbalance_sweep.py --dry-run
python scripts\make_imbalance_sweep.py
```

The dry run writes nothing and reports the five splits. The real run writes
`train_lt_*.jsonl` and `manifest_sweep.json`. It never re-derives the pinned
splits, so `manifest.json` and every digest behind the published results stay
exactly as they are.

The script then reports whether `lt_r33` is byte-identical to the pinned
`train_long_tailed`. It should be: same source, same seed, same keep rates. If
it is, that point on the curve is already measured and does not need training
again.

### Prove the chain before the long run

```powershell
python experiments\train_transformer.py --variant lt_r10 --seeds 0 --smoke
```

Two minutes on 2,000 rows for one epoch, through to the probability files. It
catches a broken environment before an overnight job does.

### The sweep

```powershell
python experiments\train_transformer.py --variant lt_r3   --seeds 0 1 2 3 4
python experiments\train_transformer.py --variant lt_r10  --seeds 0 1 2 3 4
python experiments\train_transformer.py --variant lt_r100 --seeds 0 1 2 3 4
```

Or as one unattended job, which also leaves a log to read in the morning:

```powershell
foreach ($v in @("lt_r3","lt_r10","lt_r100")) { python experiments\train_transformer.py --variant $v --seeds 0 1 2 3 4 2>&1 | Tee-Object -Append sweep.log }
```

`lt_r1` is omitted: it is identical to `train_balanced`, already measured.
`lt_r33` is omitted if the script confirmed it matches the pinned split.
Add either back only if the script says otherwise.

### Send the results back

```powershell
git add results\probs models\manifest_*.json data\processed\manifest_sweep.json
git commit -m "Transformer probabilities for the imbalance sweep, five seeds"
git push
```

The ONNX graphs stay on the workstation. What travels is `results/probs/*.npz`,
about 300 KB each, from which every figure is recomputed on the laptop.

---

## How long it takes

Training time scales with the number of training rows. Quantisation and
prediction do not: they always score the same 13,600 test and calibration rows,
and they run on CPU, so they cost about the same twelve minutes per seed
whatever the split size.

| Variant | Ratio | Train rows | Train | Quantise and predict | Per seed | Five seeds |
|---|---|---|---|---|---|---|
| `lt_r3` | 3:1 | 72,105 | ~13 min | ~12 min | ~25 min | **~2 h 5 m** |
| `lt_r10` | 10:1 | 51,300 | ~9 min | ~12 min | ~21 min | **~1 h 45 m** |
| `lt_r100` | 100:1 | 35,910 | ~6 min | ~12 min | ~18 min | **~1 h 30 m** |

**About five and a half hours in total**, nearly all unattended. Start it in the
evening and read `sweep.log` in the morning.

If `lt_r33` has to be trained as well, add roughly another hour and a half.

---

## Back on the laptop

```bash
cd ~/Developer/quantised-calibration
git pull
python experiments/run_data_sensitivity.py
```

The analysis reads whatever probability files are present, so the sweep points
join the existing ones without further work.
