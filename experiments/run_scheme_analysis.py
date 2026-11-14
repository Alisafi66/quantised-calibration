"""Which property of the quantisation scheme costs the coverage?

`run_scheme_decomposition.py` adds three arms so that each comparison below
moves exactly one property. This reads the resulting probability files and
reports those comparisons.

Every arm is scored against the same baseline the headline result uses: a
confidence threshold fitted once on the full-precision calibration split for a
95 per cent delivered-precision target, then carried across unchanged. That is
the practitioner's position, which is the whole point of the experiment: the
threshold is chosen before quantisation and nobody refits it afterwards.

Run:  python experiments/run_scheme_analysis.py --variant long_tailed
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from qcal.metrics import (coverage_and_precision,  # noqa: E402
                          expected_calibration_error,
                          threshold_for_target_precision)

PROBS = pathlib.Path("results/probs")
SCHEME = PROBS / "scheme"
OUT = pathlib.Path("results/raw")
TARGET = 0.95

# arm -> (which file holds it, which key inside). The two published arms keep
# their original names in the original files and are not recomputed.
SOURCE = {
    "int8_static": ("published", "int8"),
    "int4_block128": ("published", "int4"),
    "int8_dynamic": ("scheme", "int8_dynamic"),
    "int8_weight_only": ("scheme", "int8_weight_only"),
    "int4_block32": ("scheme", "int4_block32"),
}

PROPERTIES = {
    "int8_static": ("8 bit", "static, calibrated", "per-channel"),
    "int8_dynamic": ("8 bit", "dynamic, per input", "per-channel"),
    "int8_weight_only": ("8 bit", "float", "block-128"),
    "int4_block128": ("4 bit", "float", "block-128"),
    "int4_block32": ("4 bit", "float", "block-32"),
}

CONTRASTS = [
    ("static calibration", "int8_dynamic", "int8_static"),
    ("activation quantisation", "int8_weight_only", "int8_dynamic"),
    ("bit width", "int4_block128", "int8_weight_only"),
    ("weight granularity", "int4_block32", "int4_block128"),
]


def load(variant: str, seed: int, split: str, which: str):
    path = ((PROBS if which == "published" else SCHEME)
            / f"{variant}_seed{seed}_{split}.npz")
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def conf_corr(prob: np.ndarray, label: np.ndarray):
    return prob.max(1), (prob.argmax(1) == label).astype(float)


def arm_series(variant: str, arm: str, seeds: list[int]):
    """Per-seed coverage, ECE and accuracy for one arm under the carried
    threshold, plus the FP32 coverage of the same seed for pairing."""
    which, key = SOURCE[arm]
    cov, ece, acc, cov_fp32 = [], [], [], []
    for seed in seeds:
        base_cal = load(variant, seed, "prob_calib", "published")
        base_test = load(variant, seed, "test", "published")
        th = threshold_for_target_precision(
            *conf_corr(base_cal["fp32"], base_cal["label"]), TARGET)

        src = load(variant, seed, "test", which)
        prob, label = src[key], src["label"]
        if not np.array_equal(label, base_test["label"]):
            raise SystemExit(
                f"{variant} seed {seed}: the {which} file's labels do not match "
                f"the published file's. The two were written from different row "
                f"orders, so pairing them would be meaningless.")
        c, _ = coverage_and_precision(*conf_corr(prob, label), th)
        cov.append(100 * c)
        ece.append(expected_calibration_error(*conf_corr(prob, label)))
        acc.append(float((prob.argmax(1) == label).mean()))

        f, _ = coverage_and_precision(*conf_corr(base_test["fp32"], label), th)
        cov_fp32.append(100 * f)
    return (np.array(cov), np.array(ece), np.array(acc), np.array(cov_fp32))


def paired(d: np.ndarray) -> dict:
    t = stats.ttest_1samp(d, 0.0)
    same = int(max((d > 0).sum(), (d < 0).sum()))
    return dict(mean=float(d.mean()), t=float(t.statistic), p=float(t.pvalue),
                same_sign=f"{same}/{len(d)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="long_tailed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    variant, seeds = args.variant, args.seeds
    t_crit = stats.t.ppf(0.975, len(seeds) - 1)

    series, missing = {}, []
    for arm in SOURCE:
        try:
            series[arm] = arm_series(variant, arm, seeds)
        except FileNotFoundError as exc:
            missing.append((arm, exc.args[0]))

    print(f"{'=' * 76}\n{variant.upper().replace('_', '-')}   coverage at a "
          f"{TARGET:.0%} delivered-precision target, {len(seeds)} seeds\n"
          f"{'=' * 76}")
    if missing:
        print("\nnot yet measured:")
        for arm, path in missing:
            print(f"  {arm:17s} needs {path}")

    fp32 = next(iter(series.values()))[3] if series else None
    if fp32 is not None:
        print(f"\n  {'arm':17s} {'bits':>5s} {'activations':>19s} "
              f"{'weights':>11s} {'coverage':>9s} {'vs FP32':>9s} {'t':>7s} "
              f"{'ECE':>6s}")
        print(f"  {'fp32':17s} {'32':>5s} {'float':>19s} {'float':>11s} "
              f"{fp32.mean():8.2f}% {'':>9s} {'':>7s}")
        for arm, (cov, ece, acc, base) in series.items():
            st = paired(cov - base)
            bits, act, grain = PROPERTIES[arm]
            print(f"  {arm:17s} {bits:>5s} {act:>19s} {grain:>11s} "
                  f"{cov.mean():8.2f}% {st['mean']:+8.2f}pp {st['t']:+7.2f} "
                  f"{ece.mean():6.4f}")

    print(f"\n{'=' * 76}\nONE PROPERTY AT A TIME   (t critical at "
          f"{len(seeds) - 1} df is {t_crit:.3f})\n{'=' * 76}")
    results = {}
    for prop, arm, ref in CONTRASTS:
        if arm not in series or ref not in series:
            print(f"\n  {prop}: needs both {arm} and {ref}")
            continue
        d = series[arm][0] - series[ref][0]
        st = paired(d)
        verdict = ("this property accounts for the difference"
                   if abs(st["t"]) > t_crit and abs(st["mean"]) > 0.5
                   else "no detectable effect from this property")
        print(f"\n  {prop}")
        print(f"    {arm} against {ref}")
        print(f"    {series[arm][0].mean():6.2f}% against "
              f"{series[ref][0].mean():6.2f}%   {st['mean']:+6.2f}pp   "
              f"t({len(seeds) - 1}) = {st['t']:+6.2f}   "
              f"signs {st['same_sign']}")
        print(f"    {verdict}")
        results[prop] = dict(arm=arm, reference=ref,
                             arm_coverage=float(series[arm][0].mean()),
                             reference_coverage=float(series[ref][0].mean()),
                             **st)

    if series:
        print(f"\n{'=' * 76}\nECE ACROSS EVERY ARM   the instrument RQ1 "
              f"questions\n{'=' * 76}")
        base_ece = [expected_calibration_error(
            *conf_corr(load(variant, s, "test", "published")["fp32"],
                       load(variant, s, "test", "published")["label"]))
            for s in seeds]
        print(f"  {'fp32':17s} {np.mean(base_ece):6.4f}")
        for arm, (cov, ece, acc, base) in series.items():
            st = paired(ece - np.array(base_ece))
            flag = "detected" if abs(st["t"]) > t_crit else "blind"
            print(f"  {arm:17s} {ece.mean():6.4f}  {st['mean']:+7.4f} "
                  f"t = {st['t']:+6.2f}  {flag}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"scheme_decomposition_{variant}.json"
    path.write_text(json.dumps(dict(
        variant=variant, seeds=seeds, target=TARGET,
        arms={a: dict(coverage=float(s[0].mean()), ece=float(s[1].mean()),
                      accuracy=float(s[2].mean()),
                      vs_fp32=paired(s[0] - s[3]))
              for a, s in series.items()},
        contrasts=results), indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
