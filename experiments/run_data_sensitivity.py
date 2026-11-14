"""Is the RQ2 result a property of the model, or of how the data was built?

The long-tailed split is a controlled subsample of a balanced corpus at keep
rates 1.0 / 0.3 / 0.1 / 0.03. If the coverage loss under INT8 is an artefact of
that construction, it should live entirely in the classes the subsample starved
and vanish once they are removed. If it is a property of what quantisation does
to the confidence ordering, it should survive their removal.

This runs on the committed probability files and needs no GPU, so it is the
cheapest available answer to the question, and it decides what the second round
of experiments has to test.

Run:  python experiments/run_data_sensitivity.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from importlib import util

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from qcal.data import AG_NEWS_CLASSES  # noqa: E402
from qcal.metrics import (coverage_and_precision,  # noqa: E402
                          threshold_for_target_precision)

PROBS = pathlib.Path("results/probs")
PROC_MANIFEST = pathlib.Path("data/processed/manifest_sweep.json")
OUT = pathlib.Path("results/raw")
SEEDS = [0, 1, 2, 3, 4]
TARGET = 0.95

# The keep rate applied to each class when the long tail was drawn, and the
# training rows that survived it.
KEEP = {0: 1.00, 1: 0.30, 2: 0.10, 3: 0.03}
TRAIN_ROWS = {0: 28_500, 1: 8_550, 2: 2_850, 3: 855}


def conf_corr(npz, precision, mask=None):
    """Confidence and correctness, optionally restricted to a subset of rows."""
    pr = npz[precision]
    label = npz["label"]
    if mask is not None:
        pr, label = pr[mask], label[mask]
    return pr.max(1), (pr.argmax(1) == label).astype(float)


def load(variant, seed):
    return (np.load(PROBS / f"{variant}_seed{seed}_test.npz"),
            np.load(PROBS / f"{variant}_seed{seed}_prob_calib.npz"))


def carried_threshold(cal):
    """The practitioner's threshold: fitted once at full precision and carried
    across unchanged. This is the comparison the headline result reports."""
    return threshold_for_target_precision(*conf_corr(cal, "fp32"), TARGET)


def paired(deltas):
    """Mean, t and the sign agreement across seeds. With five observations a
    consistent sign carries as much weight as the statistic."""
    d = np.asarray(deltas, dtype=float)
    t = stats.ttest_1samp(d, 0.0)
    same = int(max((d > 0).sum(), (d < 0).sum()))
    return dict(mean=float(d.mean()), t=float(t.statistic),
                p=float(t.pvalue), same_sign=f"{same}/{len(d)}")


def coverage_by_subset(variant, precision, mask_fn):
    """Coverage under the carried threshold, per seed, over a row subset."""
    out = []
    for seed in SEEDS:
        test, cal = load(variant, seed)
        th = carried_threshold(cal)
        mask = mask_fn(test["label"])
        cov, _ = coverage_and_precision(*conf_corr(test, precision, mask), th)
        out.append(100 * cov)
    return out


def headline(variant):
    """Reproduce the reported figure, as a check that nothing has drifted.

    Both quantised arms are compared against full precision, because the
    thesis makes two claims and the sweep tests the generality of each: that
    INT8 costs coverage, and that INT4 does not.
    """
    rows = {}
    for precision in ("fp32", "int8", "int4"):
        rows[precision] = coverage_by_subset(variant, precision,
                                             lambda lab: np.ones(len(lab),
                                                                 bool))
    stats_ = {p: paired([q - f for q, f in zip(rows[p], rows["fp32"])])
              for p in ("int8", "int4")}
    return rows, stats_


def per_class(variant):
    """Where the answered inputs are lost, class by class."""
    out = {}
    for c in range(len(AG_NEWS_CLASSES)):
        fp = coverage_by_subset(variant, "fp32", lambda lab, c=c: lab == c)
        i8 = coverage_by_subset(variant, "int8", lambda lab, c=c: lab == c)
        out[c] = dict(fp32=float(np.mean(fp)), int8=float(np.mean(i8)),
                      **paired([a - b for a, b in zip(i8, fp)]))
    return out


def drop_rarest(variant):
    """The decisive test. Remove the starved classes from the test set and ask
    whether the effect is still there among the classes that were not starved."""
    out = {}
    for n_drop in (0, 1, 2):
        keep_classes = sorted(KEEP, key=KEEP.get, reverse=True)[
            :len(KEEP) - n_drop]

        def mask_fn(lab, kc=tuple(keep_classes)):
            return np.isin(lab, kc)

        fp = coverage_by_subset(variant, "fp32", mask_fn)
        i8 = coverage_by_subset(variant, "int8", mask_fn)
        label = ('all four classes' if n_drop == 0 else
                 f'rarest {n_drop} class{"es" if n_drop > 1 else ""} removed')
        out[label] = dict(
            classes=[AG_NEWS_CLASSES[c] for c in sorted(keep_classes)],
            train_rows=sum(TRAIN_ROWS[c] for c in keep_classes),
            fp32=float(np.mean(fp)), int8=float(np.mean(i8)),
            **paired([a - b for a, b in zip(i8, fp)]))
    return out


def sweep_ratios():
    """The imbalance ratio of every sweep split, from the sweep's own table.

    The ratio is a declared constant of the experiment, so it is read from the
    script that defines the splits rather than from a generated manifest. The
    manifest is the stronger record where it is present, because it also
    carries the digest of the split actually written, but `data/processed` is
    not committed, so on a machine that only received the probability files
    the table below is the sole surviving source.
    """
    out = {}
    src = pathlib.Path("scripts/make_imbalance_sweep.py")
    if src.exists():
        spec = util.spec_from_file_location("_sweep", src)
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name, keep in mod.SWEEP.items():
            out[name] = max(keep) / min(keep)
    return out


def available_variants():
    """Whatever has been produced, in ratio order where the ratio is known.

    The dose-response table orders itself as new points arrive, so a partial
    sweep still reports the points it has.
    """
    seen = sorted({f.name.split("_seed")[0]
                   for f in PROBS.glob("*_seed*_test.npz")})
    ratios = {"balanced": 1.0, "long_tailed": 100 / 3, **sweep_ratios()}
    if PROC_MANIFEST.exists():
        for name, rec in json.loads(PROC_MANIFEST.read_text())["splits"].items():
            ratios[name.removeprefix("train_")] = rec["imbalance_ratio"]
    return [(v, ratios.get(v)) for v in
            sorted(seen, key=lambda v: (ratios.get(v) is None,
                                        ratios.get(v) or 0))]


def dose_response(variants):
    """Coverage loss against imbalance ratio: the point of the second round."""
    print(f"\n{'=' * 68}\nDOSE RESPONSE   coverage lost to INT8 by imbalance "
          f"ratio\n{'=' * 68}")
    print(f"  {'variant':12s} {'ratio':>8s} {'FP32':>8s}   "
          f"{'INT8':>8s} {'t(4)':>7s}   {'INT4':>8s} {'t(4)':>7s}")
    rows = []
    for variant, ratio_ in variants:
        try:
            cov, stat = headline(variant)
        except FileNotFoundError:
            continue
        label = f"{ratio_:.0f}:1" if ratio_ else "?"
        print(f"  {variant:12s} {label:>8s} {np.mean(cov['fp32']):7.2f}%   "
              f"{stat['int8']['mean']:+7.2f}pp {stat['int8']['t']:+7.2f}   "
              f"{stat['int4']['mean']:+7.2f}pp {stat['int4']['t']:+7.2f}")
        rows.append(dict(variant=variant, ratio=ratio_,
                         fp32=float(np.mean(cov["fp32"])),
                         int8=float(np.mean(cov["int8"])),
                         int4=float(np.mean(cov["int4"])),
                         **{f"delta_{p}": stat[p] for p in ("int8", "int4")}))
    if len(rows) < 3:
        print("\n  Only one or two points so far. The curve needs the sweep;"
              "\n  see docs/round-two-runbook.md.")
    return rows


def report():
    result = {}
    variants = available_variants()
    for variant, _ in variants:
        print(f"\n{'=' * 68}\n{variant.upper().replace('_', '-')}"
              f"   coverage at a {TARGET:.0%} delivered-precision target")
        print("=" * 68)

        rows, stat = headline(variant)
        print(f"\nheadline, all classes")
        print(f"  fp32  {np.mean(rows['fp32']):6.2f}%")
        for p in ("int8", "int4"):
            print(f"  {p:5s} {np.mean(rows[p]):6.2f}%   "
                  f"{stat[p]['mean']:+.2f} points against FP32  "
                  f"t(4) = {stat[p]['t']:+.2f}  signs {stat[p]['same_sign']}")

        if variant not in ("long_tailed", "balanced"):
            result[variant] = dict(headline=stat)
            continue

        print(f"\nby true class, FP32 against INT8")
        print(f"  {'class':10s} {'train':>7s} {'FP32':>7s} {'INT8':>7s} "
              f"{'change':>8s} {'t(4)':>7s}  signs")
        pc = per_class(variant)
        for c, r in pc.items():
            print(f"  {AG_NEWS_CLASSES[c]:10s} {TRAIN_ROWS[c]:7,d} "
                  f"{r['fp32']:6.2f}% {r['int8']:6.2f}% "
                  f"{r['mean']:+7.2f}pp {r['t']:+7.2f}  {r['same_sign']}")

        print(f"\nremoving the starved classes from the test set")
        dr = drop_rarest(variant)
        for label, r in dr.items():
            print(f"  {label:28s} {r['fp32']:6.2f}% -> {r['int8']:6.2f}%  "
                  f"{r['mean']:+6.2f}pp  t(4) = {r['t']:+6.2f}  "
                  f"signs {r['same_sign']}")

        result[variant] = dict(headline=stat, per_class=pc,
                               drop_rarest=dr)

    result["dose_response"] = dose_response(variants)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "data_sensitivity.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {path}")
    return result


if __name__ == "__main__":
    report()
