"""Derive additional long-tailed training splits at other imbalance ratios.

The reported effect rests on one long tail, drawn at keep rates
1.0 / 0.3 / 0.1 / 0.03. That is a chosen parameter, and a result that holds
only at the parameter chosen is an artefact. This writes the same split at
other ratios so the question can be answered as a dose-response rather than a
single replication.

It is additive by construction. It reads `data/processed/train_balanced.jsonl`
and writes only new `train_lt_*.jsonl` files alongside a separate manifest, so
the pinned splits and `manifest.json` behind the published results are never
re-derived and their digests cannot move.

Run:  python scripts/make_imbalance_sweep.py
      python scripts/make_imbalance_sweep.py --dry-run
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
from qcal.data import (AG_NEWS_CLASSES, long_tail_indices,  # noqa: E402
                       read_jsonl, write_jsonl)

PROC = pathlib.Path("data/processed")
MANIFEST = PROC / "manifest_sweep.json"

# name -> per-class keep rates. `r33` is the published split, regenerated here
# under its sweep name so that every point on the curve is produced by one
# code path rather than one point coming from elsewhere.
SWEEP = {
    "lt_r1":   [1.00, 1.00, 1.00, 1.00],   # balanced, the floor of the curve
    "lt_r3":   [1.00, 0.70, 0.50, 0.33],   # mild
    "lt_r10":  [1.00, 0.50, 0.20, 0.10],   # moderate
    "lt_r33":  [1.00, 0.30, 0.10, 0.03],   # the published split
    "lt_r100": [1.00, 0.20, 0.05, 0.01],   # severe
}


def ratio(keep):
    return max(keep) / min(keep)


def main(dry_run: bool) -> None:
    cfg = yaml.safe_load(open("configs/default.yaml"))
    seed = cfg["seed_list"][0]

    src = PROC / "train_balanced.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} not found; run scripts/fetch_data.py first")
    train = read_jsonl(src, "train_balanced")

    published = cfg["dataset"]["long_tail_fractions"]
    if SWEEP["lt_r33"] != published:
        raise SystemExit(
            f"lt_r33 is {SWEEP['lt_r33']} but the config says {published}; "
            f"the sweep must contain the published split exactly")

    manifest = {
        "derived_from": "train_balanced.jsonl",
        "derived_utc": dt.datetime.now(dt.timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "classes": AG_NEWS_CLASSES,
        "note": "Additive. The pinned splits and manifest.json are untouched.",
        "splits": {},
    }

    print(f"source: {src}  n={len(train):,}\n")
    for name, keep in SWEEP.items():
        idx = long_tail_indices(train.label, keep, seed)
        label = train.label[idx]
        counts = {AG_NEWS_CLASSES[i]: int((label == i).sum())
                  for i in range(len(AG_NEWS_CLASSES))}
        line = (f"  {name:8s} ratio {ratio(keep):6.1f}:1  "
                f"n={len(idx):>7,}  {counts}")
        if dry_run:
            print(line + "   [dry run]")
            continue
        digest = write_jsonl(PROC / f"train_{name}.jsonl",
                             [train.text[i] for i in idx], label)
        manifest["splits"][f"train_{name}"] = {
            "n": len(idx), "sha256": digest, "keep_fractions": keep,
            "imbalance_ratio": ratio(keep), "class_counts": counts}
        print(line)

    if dry_run:
        print("\nnothing written")
        return

    # lt_r33 should reproduce the pinned long tail byte for byte: same source,
    # same seed, same keep rates. If it does, the published probabilities can
    # stand as that point on the curve and need not be recomputed.
    pinned = json.loads((PROC / "manifest.json").read_text())
    want = pinned["splits"]["train_long_tailed"]["sha256"]
    got = manifest["splits"]["train_lt_r33"]["sha256"]
    manifest["lt_r33_matches_pinned"] = (want == got)
    print()
    if want == got:
        print("  lt_r33 is byte-identical to the pinned train_long_tailed.")
        print("  SKIP it when training: reuse results/probs/long_tailed_seed*.")
    else:
        print("  WARNING: lt_r33 does not match the pinned train_long_tailed.")
        print(f"    pinned {want[:16]}  sweep {got[:16]}")
        print("  Train lt_r33 as well, and find out why before trusting the")
        print("  curve, because the two should be the same split.")

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {MANIFEST}")
    print("\nNext, on the CUDA machine:")
    for name in SWEEP:
        if name == "lt_r1":
            print(f"  {name:8s} skip, identical to train_balanced")
            continue
        if name == "lt_r33" and manifest.get("lt_r33_matches_pinned"):
            print(f"  {name:8s} skip, identical to the pinned long tail")
            continue
        print(f"  python experiments/train_transformer.py "
              f"--variant {name} --seeds 0 1 2 3 4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report the splits that would be written, write none")
    main(ap.parse_args().dry_run)
