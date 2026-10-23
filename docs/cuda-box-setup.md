# Running on the CUDA workstation

Fine-tuning is the only stage that needs the RTX 4080. Everything after it,
including INT4, runs on CPU through ONNX Runtime, so the laptop can carry the
quantisation, evaluation, analysis and writing.

## What runs where

| Stage | Machine | Why |
|---|---|---|
| Fine-tune DistilBERT | RTX 4080 | ~20 min on GPU against several hours on MPS |
| Export to ONNX | either | CPU operation, seconds |
| INT8 and INT4 quantisation | either | ONNX Runtime, CPU only |
| Evaluation, analysis, figures | MacBook | no GPU involved |
| Device measurement | Android handset | RQ3 |

`bitsandbytes` is **not** required. The outline's earlier claim that INT4 needs
CUDA was wrong: ONNX Runtime's `MatMulNBitsQuantizer` does weight-only
block-wise INT4 on CPU, and it is the better choice anyway because ONNX Runtime
Mobile is what the handset will run. One toolchain from desktop to device.

## Setup on the workstation

```powershell
git clone https://github.com/Alisafi66/quantised-calibration.git
cd quantised-calibration
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True NVIDIA GeForce RTX 4080 SUPER`. If it prints
`False`, the CPU build of torch was installed; uninstall it and repeat the
indexed install.

Then prove the chain before committing the GPU to anything long:

```powershell
python scripts/fetch_data.py --config configs/default.yaml
python experiments/train_transformer.py --smoke
```

`--smoke` trains on 2,000 rows for one epoch and runs the whole chain through
to the probability files. It takes a few minutes and catches a broken CUDA
install before an overnight run does.

Then the real thing:

```powershell
python experiments/train_transformer.py --variant balanced --seeds 0 1 2 3 4
python experiments/train_transformer.py --variant long_tailed --seeds 0 1 2 3 4
```

## What crosses between machines

**Not the models.** A fine-tuned graph is 256 MB and the full grid comes to
7.5 GB, which exceeds GitHub's 100 MB per-file limit and has no business in a
repository. `models/` is gitignored.

What travels is `results/probs/*.npz`: the predicted probabilities for every
precision on the test and calibration splits, about 300 KB each. Every figure
in RQ1 and RQ2 is recomputed from them, calibration error, thresholds,
coverage, conformal sets and temperature re-fitting alike, so the analysis runs
on the laptop without the graphs ever leaving this machine.

```powershell
git add results/probs models/manifest_*.json
git commit -m "Transformer probabilities, balanced and long-tailed, five seeds"
git push
```

The ONNX graphs stay here until RQ3, when they go to the handset directly.

## Expected timings

| | RTX 4080 | MacBook M4 (MPS) |
|---|---|---|
| Fine-tune, 114k rows, 2 epochs | ~20 min | ~4 h |
| Export to ONNX | 30 s | 30 s |
| INT8 static quantisation | 1 min | 1 min |
| INT4 block quantisation | 10 s | 10 s |
| Predict 13,600 rows x 3 precisions | ~10 min | ~25 min |

Prediction is CPU-bound through ONNX Runtime on both machines, and INT4 is the
slowest of the three because every matrix multiplication dequantises its
weights first. Five seeds on each variant is therefore roughly five hours in
total, most of it unattended.

If CUDA is unavailable, `--smoke` still works on CPU and the whole pipeline can
be exercised; only the full fine-tune is impractical without a GPU.
