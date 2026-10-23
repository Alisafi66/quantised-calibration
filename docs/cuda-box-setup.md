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

Then pin the data and train:

```powershell
python scripts/fetch_data.py --config configs/default.yaml
python experiments/train_transformer.py --variant balanced --seeds 0 1 2 3 4
```

Checkpoints are **not** committed. Push the exported ONNX graphs and the
results JSON; the Mac pulls those and does the analysis.

## Expected timings

| | RTX 4080 | MacBook M4 (MPS) |
|---|---|---|
| Fine-tune, 102k rows, 2 epochs | ~20 min | ~4 h |
| Export to ONNX | 30 s | 30 s |
| INT8 static quantisation | 1 min | 1 min |
| INT4 block quantisation | 10 s | 10 s |
| Evaluate 7,600 rows | 1 min | 1 min |

Five seeds on the balanced split and five on the long-tailed is therefore
roughly three hours of GPU time, not days.
