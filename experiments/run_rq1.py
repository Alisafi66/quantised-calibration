"""RQ1 — does post-training quantisation distort confidence, apart from accuracy?

Every cell records macro-F1 beside every calibration figure. Accuracy is an
observed covariate here, never a filter: a precision whose accuracy falls
further is interpreted rather than excluded, which is what keeps INT4 in the
analysis instead of discarding the condition the thesis is named after.

Run:  python experiments/run_rq1.py --pilot          # one seed, subsampled
      python experiments/run_rq1.py                  # the full grid
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import yaml
from sklearn.metrics import f1_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from qcal.data import AG_NEWS_CLASSES, read_jsonl  # noqa: E402
from qcal.metrics import (aurc, bootstrap_ci, expected_calibration_error,  # noqa: E402
                          per_class_ece)
from qcal.models import TfidfLinear  # noqa: E402

PROC = pathlib.Path("data/processed")
OUT = pathlib.Path("results/raw")


def evaluate(model, text, label, precision, act_sample, n_bins, n_boot):
    """One cell of the grid: a model at a precision, measured every way RQ1 asks.

    Intervals come from resampling the evaluation set rather than from the
    seed list.  The linear baseline is deterministic, so every seed returns
    the same number and seed spread would report an uncertainty of zero that
    no reader should believe.
    """
    proba = model.predict_proba(text, precision, act_sample)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    correct = (pred == label).astype(float)
    return {
        "precision": precision,
        "macro_f1": float(f1_score(label, pred, average="macro")),
        "accuracy": float(correct.mean()),
        "ece": expected_calibration_error(conf, correct, n_bins),
        "aurc": aurc(conf, correct),
        "mean_confidence": float(conf.mean()),
        "ece_ci95": bootstrap_ci(lambda c, k: expected_calibration_error(c, k, n_bins),
                                 conf, correct, n_resamples=n_boot),
        "aurc_ci95": bootstrap_ci(aurc, conf, correct, n_resamples=n_boot),
        "per_class_ece": {AG_NEWS_CLASSES[c]: v for c, v in
                          per_class_ece(conf, correct, label,
                                        len(AG_NEWS_CLASSES), n_bins).items()},
    }


def main(cfg: dict, pilot: bool, variant: str | None = None) -> None:
    variant = variant or cfg["dataset"]["variant"]
    train_name = "train_long_tailed" if variant == "long_tailed" else "train_balanced"
    n_bins = cfg["evaluation"]["ece_bins"]
    n_boot = cfg["evaluation"]["bootstrap_resamples"]
    seeds = cfg["seed_list"][:1] if pilot else cfg["seed_list"]
    precisions = cfg["quantisation"]["precisions"]

    train = read_jsonl(PROC / f"{train_name}.jsonl", train_name)
    test = read_jsonl(PROC / "test.jsonl", "test")
    ptq = read_jsonl(PROC / "ptq_calib.jsonl", "ptq_calib")

    if pilot:
        # a pilot exists to prove the pipeline closes, not to produce a result
        rng = np.random.default_rng(0)
        keep = rng.choice(len(train.text), size=min(20_000, len(train.text)), replace=False)
        train = type(train)([train.text[i] for i in keep], train.label[keep], "pilot_train")
        keep = rng.choice(len(test.text), size=2_000, replace=False)
        test = type(test)([test.text[i] for i in keep], test.label[keep], "pilot_test")

    print(f"variant={variant}  train={len(train):,}  test={len(test):,}  "
          f"seeds={seeds}  precisions={precisions}")

    rows = []
    for seed in seeds:
        t0 = time.time()
        model = TfidfLinear(seed=seed).fit(train.text, train.label)
        # the activation range is observed on the PTQ sample, never on test
        act_sample = model.features(ptq.text)
        print(f"  seed {seed}: fitted in {time.time() - t0:.1f}s "
              f"({model.features(train.text[:1]).shape[1]:,} features)")
        for precision in precisions:
            r = evaluate(model, test.text, test.label, precision, act_sample,
                         n_bins, n_boot)
            r["seed"] = seed
            r["variant"] = variant
            r["n_train"] = len(train)
            rows.append(r)
            lo, hi = r["ece_ci95"]
            print(f"    {precision:5s}  macro-F1 {r['macro_f1']:.4f}   "
                  f"ECE {r['ece']:.4f} [{lo:.4f}, {hi:.4f}]   "
                  f"AURC {r['aurc']:.4f}   mean conf {r['mean_confidence']:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = "pilot" if pilot else f"{variant}_tfidf"
    path = OUT / f"rq1_{stamp}.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {path}")

    # the headline RQ1 comparison, read against accuracy rather than filtered by it
    base = next(r for r in rows if r["precision"] == "fp32")
    print("\nchange from FP32 (seed 0):")
    for r in rows:
        if r["precision"] == "fp32" or r["seed"] != rows[0]["seed"]:
            continue
        print(f"  {r['precision']:5s}  dF1 {r['macro_f1'] - base['macro_f1']:+.4f}   "
              f"dECE {r['ece'] - base['ece']:+.4f}   "
              f"dAURC {r['aurc'] - base['aurc']:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pilot", action="store_true",
                    help="one seed on a subsample, to prove the pipeline closes")
    ap.add_argument("--variant", choices=["balanced", "long_tailed"],
                    help="override the split named in the config")
    args = ap.parse_args()
    main(yaml.safe_load(open(args.config)), args.pilot, args.variant)
