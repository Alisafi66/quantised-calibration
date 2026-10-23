"""Pin an AG News snapshot and derive every split the experiments use.

Reproducibility depends on knowing exactly which bytes produced a figure, so
nothing is used until it has been written to a canonical file and hashed.  The
hash goes back into the config, and a manifest records the counts so that a
reader can tell at a glance what the long-tailed variant actually did.

Run:  python scripts/fetch_data.py --config configs/default.yaml --write-config
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from qcal.data import (  # noqa: E402
    AG_NEWS_CLASSES, long_tail_indices, ptq_calibration_indices,
    read_jsonl, sha256_file, stratified_holdout, write_jsonl,
)

RAW = pathlib.Path("data/raw/ag_news")
PROC = pathlib.Path("data/processed")
HF_ID = "fancyzhx/ag_news"


def download() -> None:
    """Fetch the corpus once and write it out in a form that hashes stably."""
    if (RAW / "train.jsonl").exists() and (RAW / "test.jsonl").exists():
        print("raw snapshot already present, not re-downloading")
        return
    from datasets import load_dataset
    print(f"downloading {HF_ID} ...")
    ds = load_dataset(HF_ID)
    for split in ("train", "test"):
        rows = ds[split]
        write_jsonl(RAW / f"{split}.jsonl", list(rows["text"]),
                    np.asarray(rows["label"], dtype=np.int64))
        print(f"  {split}: {len(rows)} rows")


def build(cfg: dict) -> dict:
    """Derive the splits and return a manifest describing them."""
    seed = cfg["seed_list"][0]
    calib_fraction = cfg["dataset"].get("prob_calib_fraction", 0.15)
    keep = cfg["dataset"]["long_tail_fractions"]
    n_ptq = cfg["quantisation"]["calibration_samples"]

    train_all = read_jsonl(RAW / "train.jsonl", "train_all")
    test = read_jsonl(RAW / "test.jsonl", "test")

    # probability calibration is held out of training entirely
    tr_idx, cal_idx = stratified_holdout(
        train_all.text, train_all.label, calib_fraction, seed)

    def take(idx, name):
        return ([train_all.text[i] for i in idx], train_all.label[idx], name)

    manifest: dict = {
        "corpus": HF_ID,
        "classes": AG_NEWS_CLASSES,
        "pinned_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "prob_calib_fraction": calib_fraction,
        "long_tail_fractions": keep,
        "ptq_calibration_samples": n_ptq,
        "splits": {},
        "raw_sha256": {
            "train": sha256_file(RAW / "train.jsonl"),
            "test": sha256_file(RAW / "test.jsonl"),
        },
    }

    def record(name, text, label):
        digest = write_jsonl(PROC / f"{name}.jsonl", text, label)
        counts = {AG_NEWS_CLASSES[i]: int((label == i).sum())
                  for i in range(len(AG_NEWS_CLASSES))}
        manifest["splits"][name] = {"n": len(text), "sha256": digest, "class_counts": counts}
        print(f"  {name:22s} n={len(text):>7,}  {counts}")

    print("deriving splits ...")
    tr_text, tr_label, _ = take(tr_idx, "train")
    record("train_balanced", tr_text, tr_label)

    cal_text, cal_label, _ = take(cal_idx, "prob_calib")
    record("prob_calib", cal_text, cal_label)

    record("test", test.text, test.label)

    # the long tail is drawn from the balanced training split, never from test
    lt = long_tail_indices(tr_label, keep, seed)
    record("train_long_tailed", [tr_text[i] for i in lt], tr_label[lt])

    # PTQ calibration sample, taken from training data, balanced by class
    ptq = ptq_calibration_indices(tr_label, n_ptq, seed)
    record("ptq_calib", [tr_text[i] for i in ptq], tr_label[ptq])

    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--write-config", action="store_true",
                    help="record the snapshot date and hash back into the config")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    download()
    manifest = build(cfg)

    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest written to {PROC / 'manifest.json'}")

    if args.write_config:
        text = pathlib.Path(args.config).read_text()
        text = text.replace("snapshot_date: null",
                            f"snapshot_date: {manifest['pinned_utc'][:10]}")
        text = text.replace("snapshot_sha256: null",
                            f"snapshot_sha256: {manifest['raw_sha256']['train'][:16]}")
        pathlib.Path(args.config).write_text(text)
        print(f"config updated: {args.config}")
