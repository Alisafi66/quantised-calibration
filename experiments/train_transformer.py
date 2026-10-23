"""Fine-tune the compact transformer, quantise it, and emit its predictions.

This is the only stage that wants a GPU, so it is a script of its own: run it
on the CUDA machine, push what it produces, and every later stage reads that
on whatever machine is convenient.

What crosses between machines is the probability matrix, not the model.  A
single fine-tuned graph is 256 MB and ten seeds across both variants come to
7.5 GB, which no repository should carry and GitHub would refuse anyway.  The
test-set probabilities are 122 KB and every figure in RQ1 and RQ2 can be
recomputed from them: calibration error, thresholds, coverage, conformal sets,
temperature re-fitting.  The graphs stay on the machine that made them, and go
to the handset directly when RQ3 needs them.

Run:  python experiments/train_transformer.py --variant balanced --seeds 0
      python experiments/train_transformer.py --variant balanced --seeds 0 1 2 3 4
      python experiments/train_transformer.py --smoke        # 2 min, proves the chain
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from qcal.data import read_jsonl  # noqa: E402
from qcal.onnx_quant import (EncodedReader, model_size_mb, quantise_int4,  # noqa: E402
                             quantise_int8)
from qcal.onnx_quant import run_session  # noqa: E402
from qcal.transformer import CompactTransformer, pick_device  # noqa: E402

PROC = pathlib.Path("data/processed")
MODELS = pathlib.Path("models")
PROBS = pathlib.Path("results/probs")


def build_one(cfg: dict, variant: str, seed: int, smoke: bool) -> dict:
    """Train, export and quantise a single seed. Returns a record of what it made."""
    train_name = "train_long_tailed" if variant == "long_tailed" else "train_balanced"
    train = read_jsonl(PROC / f"{train_name}.jsonl", train_name)
    ptq = read_jsonl(PROC / "ptq_calib.jsonl", "ptq_calib")

    mcfg = cfg["model"]
    tcfg = cfg.get("training", {})
    epochs = 1 if smoke else tcfg.get("epochs", 2)
    if smoke:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(train.text), size=min(2_000, len(train.text)), replace=False)
        train = type(train)([train.text[i] for i in keep], train.label[keep], "smoke")

    out = MODELS / f"{variant}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    model = CompactTransformer(model_name=mcfg["transformer_name"],
                               max_length=mcfg["max_length"], seed=seed)
    print(f"[{variant} seed {seed}] device={model.device}  train={len(train):,}  epochs={epochs}")

    t0 = time.time()
    model.fit(train.text, train.label, epochs=epochs,
              batch_size=tcfg.get("batch_size", 32), lr=tcfg.get("learning_rate", 5e-5))
    train_seconds = time.time() - t0
    print(f"  fine-tuned in {train_seconds / 60:.1f} min")

    fp32 = model.export_onnx(out / "fp32.onnx")
    print(f"  exported fp32   {model_size_mb(fp32):6.1f} MB")

    # the quantiser sees the PTQ sample only; test data is never involved
    n_calib = cfg["quantisation"]["calibration_samples"]
    ids, mask = model.encode(ptq.text[:n_calib])
    i8 = quantise_int8(fp32, out / "int8.onnx", EncodedReader(ids.numpy(), mask.numpy()))
    print(f"  quantised int8  {model_size_mb(i8):6.1f} MB")

    i4 = quantise_int4(fp32, out / "int4.onnx")
    print(f"  quantised int4  {model_size_mb(i4):6.1f} MB")

    emit_predictions(model, {"fp32": fp32, "int8": i8, "int4": i4}, variant, seed)

    return {
        "variant": variant,
        "seed": seed,
        "model_name": mcfg["transformer_name"],
        "max_length": mcfg["max_length"],
        "epochs": epochs,
        "n_train": len(train),
        "device": model.device,
        "train_seconds": round(train_seconds, 1),
        "smoke": smoke,
        "artefacts": {name: {"path": str(p.relative_to(MODELS)),
                             "size_mb": round(model_size_mb(p), 2)}
                      for name, p in (("fp32", fp32), ("int8", i8), ("int4", i4))},
    }


def emit_predictions(model, graphs: dict[str, pathlib.Path], variant: str, seed: int) -> None:
    """Write test-set probabilities for every precision, and the calibration split too.

    The calibration split is included because temperature re-fitting and split
    conformal prediction both need it, and RQ3 should not have to return to
    this machine to get it.
    """
    PROBS.mkdir(parents=True, exist_ok=True)
    for split in ("test", "prob_calib"):
        data = read_jsonl(PROC / f"{split}.jsonl", split)
        ids, mask = model.encode(data.text)
        ids, mask = ids.numpy(), mask.numpy()
        out: dict[str, np.ndarray] = {"label": data.label}
        for precision, graph in graphs.items():
            logits = run_session(graph, ids, mask)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            out[precision] = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        path = PROBS / f"{variant}_seed{seed}_{split}.npz"
        np.savez_compressed(path, **out)
        print(f"  probabilities   {path.name}  "
              f"({path.stat().st_size / 1024:.0f} KB, n={len(data):,})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--variant", choices=["balanced", "long_tailed"], default="balanced")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="defaults to the seed_list in the config")
    ap.add_argument("--smoke", action="store_true",
                    help="2,000 rows and one epoch, to prove the chain before committing a GPU")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seeds = args.seeds if args.seeds is not None else cfg["seed_list"]
    if args.smoke:
        seeds = seeds[:1]

    if pick_device() == "cpu":
        print("WARNING: no GPU found. Fine-tuning on CPU will take hours; "
              "use --smoke to check the chain, and train on the CUDA machine.\n")

    records, t0 = [], time.time()
    for seed in seeds:
        records.append(build_one(cfg, args.variant, seed, args.smoke))

    MODELS.mkdir(parents=True, exist_ok=True)
    manifest = MODELS / f"manifest_{args.variant}{'_smoke' if args.smoke else ''}.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    print(f"\n{len(records)} seed(s) in {(time.time() - t0) / 60:.1f} min")
    print(f"manifest: {manifest}")
    print("\nPush results/probs/ and this manifest. The graphs in models/ stay here; "
          "they are too large for the repository and only RQ3 needs them, on the handset.")


if __name__ == "__main__":
    main()
