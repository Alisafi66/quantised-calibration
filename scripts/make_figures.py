"""Regenerate every figure from the committed probability files.

Nothing here is hand-drawn and nothing is cached: each figure is recomputed
from `results/probs/*.npz`, so a figure in the dissertation can always be
traced back to the pinned snapshot that produced it.

Run:  python scripts/make_figures.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from qcal.data import AG_NEWS_CLASSES  # noqa: E402
from qcal.metrics import (aurc, coverage_and_precision,  # noqa: E402
                          expected_calibration_error, per_class_ece,
                          risk_coverage_curve, threshold_for_target_precision)

PROBS = pathlib.Path("results/probs")
OUT = pathlib.Path("results/figures")
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["balanced", "long_tailed"]
PRECISIONS = ["fp32", "int8", "int4"]

# Colour-blind safe, and distinguishable when printed in greyscale because the
# three differ in lightness as well as hue.
COLOUR = {"fp32": "#333333", "int8": "#C44E52", "int4": "#4C72B0"}
STYLE = {"fp32": "-", "int8": "--", "int4": ":"}
LABEL = {"fp32": "FP32", "int8": "INT8", "int4": "INT4"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def load(variant: str, seed: int):
    return (np.load(PROBS / f"{variant}_seed{seed}_test.npz"),
            np.load(PROBS / f"{variant}_seed{seed}_prob_calib.npz"))


def conf_corr(npz, precision: str):
    pr = npz[precision]
    return pr.max(1), (pr.argmax(1) == npz["label"]).astype(float)


def temp_scale(prob: np.ndarray, T: float) -> np.ndarray:
    """Softmax is shift invariant, so log(p) recovers the logits up to a constant."""
    z = np.log(np.clip(prob, 1e-12, None)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(prob: np.ndarray, y: np.ndarray) -> float:
    def nll(log_t):
        p = temp_scale(prob, np.exp(log_t))
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-2, 2), method="bounded").x))


def save(fig, name: str):
    """Write the figure and hand it back, so a notebook can display it inline."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    return fig


# --------------------------------------------------------------------------
def fig_what_ece_misses():
    """The headline: the same runs, measured two ways, giving opposite verdicts."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=False)
    jitter = {"int8": -0.08, "int4": 0.08}

    for ax, metric in zip(axes, ("ece", "coverage")):
        for vi, variant in enumerate(VARIANTS):
            for precision in ("int8", "int4"):
                deltas = []
                for seed in SEEDS:
                    test, cal = load(variant, seed)
                    if metric == "ece":
                        b = expected_calibration_error(*conf_corr(test, "fp32"))
                        q = expected_calibration_error(*conf_corr(test, precision))
                        deltas.append(q - b)
                    else:
                        cc, ck = conf_corr(cal, "fp32")
                        th = threshold_for_target_precision(cc, ck, 0.95)
                        b = coverage_and_precision(*conf_corr(test, "fp32"), th)[0]
                        q = coverage_and_precision(*conf_corr(test, precision), th)[0]
                        deltas.append(100 * (q - b))
                x = vi + jitter[precision]
                ax.scatter([x] * len(deltas), deltas, s=18, alpha=0.55,
                           color=COLOUR[precision], zorder=3)
                ax.scatter([x], [np.mean(deltas)], s=70, marker="_",
                           color=COLOUR[precision], zorder=4, linewidths=2)
        ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels(["balanced", "long-tailed"])
        ax.set_xlim(-0.45, len(VARIANTS) - 0.55)

    axes[0].set_ylabel("change in ECE")
    axes[0].set_title("Expected calibration error\nreports no effect", loc="left")
    axes[1].set_ylabel("change in coverage (pp)")
    axes[1].set_title("Coverage at a 95% precision target\nreports a large one", loc="left")

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLOUR[p],
                          label=LABEL[p], alpha=0.7) for p in ("int8", "int4")]
    axes[1].legend(handles=handles, loc="lower left", fontsize=8)
    fig.suptitle("Each point is one seed, measured against FP32 trained from the same seed",
                 fontsize=8, color="#666666", y=1.04)
    return save(fig, "01_what_ece_misses")


def fig_risk_coverage():
    """Why no threshold can help: the whole curve moves, not the operating point."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, variant in zip(axes, VARIANTS):
        for precision in PRECISIONS:
            curves = []
            for seed in SEEDS:
                test, _ = load(variant, seed)
                cov, risk = risk_coverage_curve(*conf_corr(test, precision))
                grid = np.linspace(0.02, 1.0, 300)
                curves.append(np.interp(grid, cov, risk))
            mean = np.mean(curves, axis=0)
            a = np.mean([aurc(*conf_corr(load(variant, s)[0], precision)) for s in SEEDS])
            ax.plot(grid, 100 * mean, STYLE[precision], color=COLOUR[precision],
                    linewidth=1.4, label=f"{LABEL[precision]}   AURC {a:.3f}")
            lo, hi = np.percentile(curves, [10, 90], axis=0)
            ax.fill_between(grid, 100 * lo, 100 * hi, color=COLOUR[precision], alpha=0.10)
        ax.set_title(variant.replace("_", "-"), loc="left")
        ax.set_xlabel("coverage")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("selective risk (%)")
    fig.suptitle("Risk against coverage, mean of five seeds, shaded 10th to 90th percentile",
                 fontsize=8, color="#666666", y=1.04)
    return save(fig, "02_risk_coverage")


def fig_coverage_by_target():
    """The gap widens as the promise tightens."""
    targets = [0.90, 0.95, 0.97, 0.98, 0.99]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=False)
    for ax, variant in zip(axes, VARIANTS):
        for precision in PRECISIONS:
            means, sems = [], []
            for target in targets:
                vals = []
                for seed in SEEDS:
                    test, cal = load(variant, seed)
                    cc, ck = conf_corr(cal, "fp32")
                    th = threshold_for_target_precision(cc, ck, target)
                    if np.isnan(th):
                        continue
                    vals.append(100 * coverage_and_precision(*conf_corr(test, precision), th)[0])
                means.append(np.mean(vals))
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            ax.errorbar(targets, means, yerr=sems, fmt="o",
                        color=COLOUR[precision], markersize=3.5, linewidth=1.3,
                        capsize=2, label=LABEL[precision], linestyle=STYLE[precision])
        ax.set_title(variant.replace("_", "-"), loc="left")
        ax.set_xlabel("target precision")
    axes[0].set_ylabel("coverage (%)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Threshold fitted on the FP32 calibration split and carried across unchanged",
                 fontsize=8, color="#666666", y=1.04)
    return save(fig, "03_coverage_by_target")


def fig_per_class():
    """Calibration error against how often the model saw the class."""
    sizes = {"balanced": [28500] * 4, "long_tailed": [28500, 8550, 2850, 855]}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, variant in zip(axes, VARIANTS):
        for precision in PRECISIONS:
            per_seed = []
            for seed in SEEDS:
                test, _ = load(variant, seed)
                conf, corr = conf_corr(test, precision)
                pc = per_class_ece(conf, corr, test["label"], len(AG_NEWS_CLASSES))
                per_seed.append([pc[c] for c in range(len(AG_NEWS_CLASSES))])
            ax.plot(sizes[variant], np.mean(per_seed, axis=0), STYLE[precision] + "o",
                    color=COLOUR[precision], markersize=4, linewidth=1.3,
                    label=LABEL[precision])
        ax.set_xscale("log")
        ax.set_xlabel("training examples in the class")
        ax.set_title(variant.replace("_", "-"), loc="left")
    axes[0].set_ylabel("ECE within the class")
    axes[1].legend(loc="upper right", fontsize=8)
    for i, name in enumerate(AG_NEWS_CLASSES):
        axes[1].annotate(name, (sizes["long_tailed"][i], 0), fontsize=6.5,
                         color="#777777", ha="center", va="bottom")
    fig.suptitle("Per-class calibration error, mean of five seeds",
                 fontsize=8, color="#666666", y=1.04)
    return save(fig, "04_per_class_ece")


def fig_temperature():
    """Temperature scaling repairs the calibration and not the failure."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    variant = "long_tailed"
    width = 0.35
    for ax, metric in zip(axes, ("ece", "coverage")):
        for i, precision in enumerate(("int8", "int4")):
            before, after = [], []
            for seed in SEEDS:
                test, cal = load(variant, seed)
                y, yc = test["label"], cal["label"]
                T = fit_temperature(cal[precision], yc)
                qk = (test[precision].argmax(1) == y).astype(float)
                scaled_test = temp_scale(test[precision], T)
                scaled_cal = temp_scale(cal[precision], T)
                if metric == "ece":
                    before.append(expected_calibration_error(test[precision].max(1), qk))
                    after.append(expected_calibration_error(scaled_test.max(1), qk))
                else:
                    cc, ck = conf_corr(cal, "fp32")
                    base = coverage_and_precision(
                        *conf_corr(test, "fp32"),
                        threshold_for_target_precision(cc, ck, 0.95))[0]
                    th_raw = threshold_for_target_precision(cal[precision].max(1),
                                                            (cal[precision].argmax(1) == yc).astype(float), 0.95)
                    th_new = threshold_for_target_precision(scaled_cal.max(1),
                                                            (scaled_cal.argmax(1) == yc).astype(float), 0.95)
                    before.append(100 * (coverage_and_precision(test[precision].max(1), qk, th_raw)[0] - base))
                    after.append(100 * (coverage_and_precision(scaled_test.max(1), qk, th_new)[0] - base))
            ax.bar(i - width / 2, np.mean(before), width, color=COLOUR[precision],
                   alpha=0.45, label="before" if i == 0 else None)
            ax.bar(i + width / 2, np.mean(after), width, color=COLOUR[precision],
                   label="after temperature re-fitting" if i == 0 else None)
            ax.errorbar([i - width / 2, i + width / 2],
                        [np.mean(before), np.mean(after)],
                        yerr=[np.std(before, ddof=1) / np.sqrt(5),
                              np.std(after, ddof=1) / np.sqrt(5)],
                        fmt="none", ecolor="#444444", capsize=2, linewidth=0.9)
        ax.set_xticks(range(2))
        ax.set_xticklabels([LABEL["int8"], LABEL["int4"]])
        ax.axhline(0, color="#999999", linewidth=0.8)
    axes[0].set_ylabel("ECE")
    axes[0].set_title("It repairs the calibration", loc="left")
    axes[1].set_ylabel("coverage against FP32 (pp)")
    axes[1].set_title("It does not repair the failure", loc="left")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Long-tailed split, temperature fitted on each model's own calibration split",
                 fontsize=8, color="#666666", y=1.04)
    return save(fig, "05_temperature_scaling")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    if not PROBS.exists():
        raise SystemExit("no probability files; run experiments/train_transformer.py first")
    print("writing figures to", OUT)
    for name, builder in (("01_what_ece_misses", fig_what_ece_misses),
                          ("02_risk_coverage", fig_risk_coverage),
                          ("03_coverage_by_target", fig_coverage_by_target),
                          ("04_per_class_ece", fig_per_class),
                          ("05_temperature_scaling", fig_temperature)):
        builder()
        plt.close("all")
        print(" ", name)
    print("done")
