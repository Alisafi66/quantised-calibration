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
from qcal.conformal import (class_conditional_coverage,  # noqa: E402
                            conformal_quantile, evaluate_sets,
                            lac_scores, mondrian_quantiles, predict_sets)
from qcal.metrics import (aurc, coverage_and_precision,  # noqa: E402
                          expected_calibration_error, per_class_ece,
                          risk_coverage_curve, threshold_for_target_precision)

PROBS = pathlib.Path("results/probs")
SCHEME = PROBS / "scheme"
OUT = pathlib.Path("results/figures")
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["balanced", "long_tailed"]
PRECISIONS = ["fp32", "int8", "int4"]

# Colour-blind safe, and distinguishable when printed in greyscale because the
# three differ in lightness as well as hue.
COLOUR = {"fp32": "#333333", "int8": "#C44E52", "int4": "#4C72B0"}
# Two hues, not three. The nearest third hue separates from #C44E52 by 13.7 in
# normal vision, below the floor at which a reader with full colour vision can
# tell the pair apart. The distinction the scheme figure draws is binary in any
# case, and marker shape repeats it so identity never rests on colour alone.
FIXED_SCALES = "#C44E52"
FREE_SCALES = "#4C72B0"
FAINT = "#E3B4B6"
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


def fig_conformal():
    """Conformal keeps its promise and pays for it in usefulness."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    alpha = 0.05

    # left: validity holds everywhere, efficiency does not
    width = 0.35
    for i, precision in enumerate(PRECISIONS):
        cov, sing = [], []
        for seed in SEEDS:
            test, cal = load("long_tailed", seed)
            q = conformal_quantile(lac_scores(cal[precision], cal["label"]), alpha)
            m = evaluate_sets(predict_sets(test[precision], q), test["label"])
            cov.append(100 * m["coverage"]); sing.append(100 * m["singleton_rate"])
        axes[0].bar(i - width / 2, np.mean(cov), width, color=COLOUR[precision],
                    alpha=0.45, label="coverage" if i == 0 else None)
        axes[0].bar(i + width / 2, np.mean(sing), width, color=COLOUR[precision],
                    label="answered with one label" if i == 0 else None)
    axes[0].axhline(100 * (1 - alpha), color="#444444", linewidth=0.9, linestyle="--")
    axes[0].annotate("95% target", (1.9, 100 * (1 - alpha) + 1), fontsize=7.5,
                     color="#444444", ha="right")
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels([LABEL[p] for p in PRECISIONS])
    axes[0].set_ylabel("per cent"); axes[0].set_ylim(0, 108)
    axes[0].set_title("Validity is kept, usefulness is not", loc="left")
    axes[0].legend(loc="lower left", fontsize=7.5)

    # right: pooled calibration conceals the rare class
    sizes = [28500, 8550, 2850, 855]
    for form, marker, style in (("pooled", "o", "-"), ("Mondrian", "s", "--")):
        per_class = []
        for seed in SEEDS:
            test, cal = load("long_tailed", seed)
            sc = lac_scores(cal["int8"], cal["label"])
            q = (conformal_quantile(sc, alpha) if form == "pooled"
                 else mondrian_quantiles(sc, cal["label"], alpha, 4))
            cc = class_conditional_coverage(predict_sets(test["int8"], q), test["label"], 4)
            per_class.append([100 * cc[c] for c in range(4)])
        axes[1].plot(sizes, np.mean(per_class, axis=0), marker + style,
                     color="#C44E52" if form == "pooled" else "#4C72B0",
                     markersize=4, linewidth=1.3, label=form)
    axes[1].axhline(100 * (1 - alpha), color="#444444", linewidth=0.9, linestyle=":")
    axes[1].set_xscale("log"); axes[1].set_xlabel("training examples in the class")
    axes[1].set_ylabel("coverage within the class (%)")
    axes[1].set_title("Pooled calibration conceals the rare class", loc="left")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Split conformal prediction at INT8 on the long-tailed split, five seeds",
                 fontsize=8, color="#666666", y=1.04)
    return save(fig, "06_conformal")


# Arms of the scheme decomposition, in the order of the contrast chain.
# `static` marks the arms whose activation scales are fixed from the PTQ
# sample before deployment rather than recomputed per input.
SCHEME_ARMS = [
    ("int8_static", "INT8 static, 48 matmuls", True, "published"),
    ("int8_static_weights", "INT8 static, 36 matmuls", True, "scheme"),
    ("int8_dynamic", "INT8 dynamic", False, "scheme"),
    ("int8_weight_only", "INT8 weight-only", False, "scheme"),
    ("int4_block128", "INT4 block 128", False, "published"),
    ("int4_block32", "INT4 block 32", False, "scheme"),
]
PUBLISHED_KEY = {"int8_static": "int8", "int4_block128": "int4"}


def scheme_coverage(variant: str = "long_tailed", target: float = 0.95):
    """Coverage per arm per seed under the threshold carried from full precision."""
    out = {a: [] for a, _, _, _ in SCHEME_ARMS}
    out["fp32"] = []
    for seed in SEEDS:
        test, cal = load(variant, seed)
        th = threshold_for_target_precision(*conf_corr(cal, "fp32"), target)
        c, _ = coverage_and_precision(*conf_corr(test, "fp32"), th)
        out["fp32"].append(100 * c)
        scheme = np.load(SCHEME / f"{variant}_seed{seed}_test.npz")
        for arm, _, _, src in SCHEME_ARMS:
            npz = test if src == "published" else scheme
            pr = npz[PUBLISHED_KEY.get(arm, arm)]
            c, _ = coverage_and_precision(
                pr.max(1), (pr.argmax(1) == npz["label"]).astype(float), th)
            out[arm].append(100 * c)
    return {k: np.array(v) for k, v in out.items()}


def fig_scheme_decomposition():
    """Which property of the scheme costs the coverage, and how much of it."""
    cov = scheme_coverage()
    fp32 = cov["fp32"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})

    ax = axes[0]
    ys = np.arange(len(SCHEME_ARMS))[::-1]
    for y, (arm, label, static, _) in zip(ys, SCHEME_ARMS):
        # The paired difference, because each arm is compared against the FP32
        # model trained from the same seed. Plotting the spread of absolute
        # coverage would show seed-to-seed variation that the pairing cancels,
        # and every interval would overlap despite t reaching -8.34.
        d = cov[arm] - cov["fp32"]
        m, sem = d.mean(), d.std(ddof=1) / np.sqrt(len(SEEDS))
        colour = FIXED_SCALES if static else FREE_SCALES
        ax.errorbar(m, y, xerr=sem, fmt="s" if static else "o", ms=5.5,
                    color=colour, ecolor=colour, elinewidth=1.2, capsize=2.5,
                    zorder=3)
        # A value column clear of the zero line, rather than a label trailing
        # each marker: four of the six arms sit within a point of zero and
        # their labels would otherwise collide with the FP32 rule.
        ax.text(1.35, y, f"{m:+.2f}", va="center", ha="left", fontsize=7.5,
                color="#555555")
    ax.axvline(0, color=COLOUR["fp32"], ls="--", lw=1.0, zorder=1)
    ax.text(-0.18, len(SCHEME_ARMS) - 0.62, "FP32", fontsize=7.5,
            color=COLOUR["fp32"], va="center", ha="right")
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab, _, _ in SCHEME_ARMS], fontsize=8)
    ax.set_xlabel("change in coverage against FP32 (points)")
    ax.set_xlim(-10.8, 3.6)
    ax.set_ylim(-1.5, len(SCHEME_ARMS) - 0.4)
    ax.scatter([], [], marker="s", s=28, color=FIXED_SCALES,
               label="scales fixed from the PTQ sample")
    ax.scatter([], [], marker="o", s=28, color=FREE_SCALES,
               label="scales free, or activations in float")
    ax.legend(loc="lower left", fontsize=6.8, handletextpad=0.4,
              borderpad=0.3, bbox_to_anchor=(-0.015, -0.02))
    ax.set_title("a. every arm against full precision, paired by seed",
                 fontsize=8.5, loc="left")

    ax = axes[1]
    steps = [("attention\nmatmuls",
              cov["int8_static_weights"].mean() - cov["int8_static"].mean()),
             ("scale\nsource",
              cov["int8_dynamic"].mean() - cov["int8_static_weights"].mean()),
             ("activation\nquantising",
              cov["int8_weight_only"].mean() - cov["int8_dynamic"].mean())]
    total = fp32 - cov["int8_static"].mean()
    base = cov["int8_static"].mean()
    for i, (name, d) in enumerate(steps):
        # One hue at two lightness steps: the encoding is magnitude, not
        # identity, so a second hue would imply a distinction that is not there.
        ax.bar(i, d, bottom=base, width=0.58,
               color=FIXED_SCALES if abs(d) > 0.5 * total else FAINT,
               edgecolor="white", linewidth=1.0, zorder=3)
        # A tall bar carries its label inside, where there is room; a short
        # one carries it above. The alternative puts the dominant step's text
        # across the full-precision rule.
        if d > 0.25 * total:
            ax.text(i, base + d / 2, f"{d:+.2f}\n{100 * d / total:.0f}%",
                    ha="center", va="center", fontsize=7.5, color="white",
                    linespacing=1.4, zorder=4)
        else:
            ax.text(i, base + d + 0.34, f"{d:+.2f}\n{100 * d / total:.0f}%",
                    ha="center", va="bottom", fontsize=7.5, color="#555555",
                    linespacing=1.3, zorder=4)
        if i < len(steps) - 1:
            ax.plot([i + 0.29, i + 1 - 0.29], [base + d, base + d],
                    color="#BBBBBB", lw=0.8, zorder=2)
        base += d
    ax.axhline(fp32, color=COLOUR["fp32"], ls="--", lw=1.0, zorder=1)
    ax.text(-0.55, fp32 + 0.12, "FP32", fontsize=7.5, color=COLOUR["fp32"],
            va="bottom", ha="left")
    ax.axhline(cov["int8_static"].mean(), color="#BBBBBB", lw=0.8, zorder=1)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([n for n, _ in steps], fontsize=7.5)
    ax.set_xlim(-0.6, 2.62)
    ax.set_ylabel("coverage (%)")
    ax.set_ylim(58.3, 70.8)
    ax.set_title(f"b. the {total:.2f} point gap, one property at a time",
                 fontsize=8.5, loc="left")

    fig.suptitle("Long-tailed split, five seeds, threshold carried from full "
                 "precision; bars are the standard error of the paired "
                 "difference", fontsize=8, color="#666666", y=1.02)
    fig.tight_layout()
    return save(fig, "07_scheme_decomposition")


def fig_imbalance_dose_response():
    """Coverage lost to INT8 as the training imbalance is varied."""
    from scipy import stats
    # The ratio of a split is the largest keep rate over the smallest, taken
    # from the rates in scripts/make_imbalance_sweep.py. lt_r3 is 1/0.33, not
    # 10/3: naming it r3 rounds, and fitting the rounded value moves the slope.
    points = [(1.0, "balanced"), (1 / 0.33, "lt_r3"), (1 / 0.10, "lt_r10"),
              (1 / 0.03, "long_tailed"), (1 / 0.01, "lt_r100")]
    ratios, fp32s, int8s, losses = [], [], [], []
    for ratio, variant in points:
        f, i = [], []
        for seed in SEEDS:
            test, cal = load(variant, seed)
            th = threshold_for_target_precision(*conf_corr(cal, "fp32"), 0.95)
            cf, _ = coverage_and_precision(*conf_corr(test, "fp32"), th)
            ci, _ = coverage_and_precision(*conf_corr(test, "int8"), th)
            f.append(100 * cf)
            i.append(100 * ci)
        ratios.append(ratio)
        fp32s.append(np.mean(f))
        int8s.append(np.mean(i))
        losses.append(np.mean(i) - np.mean(f))
    ratios = np.array(ratios)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axes[0]
    ax.fill_between(ratios, int8s, fp32s, color=FIXED_SCALES, alpha=0.12, lw=0)
    for series, key, mk in ((fp32s, "fp32", "o"), (int8s, "int8", "s")):
        ax.plot(ratios, series, STYLE[key], color=COLOUR[key], lw=1.8,
                marker=mk, ms=5, label=LABEL[key])
    ax.set_xscale("log")
    ax.set_xlabel("training imbalance ratio")
    ax.set_ylabel("coverage at a 95% target (%)")
    ax.set_xticks([1, 3, 10, 33, 100])
    ax.set_xticklabels(["1:1", "3:1", "10:1", "33:1", "100:1"])
    ax.set_xlim(0.78, 128)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("a. coverage falls as the tail lengthens", fontsize=8.5,
                 loc="left")

    ax = axes[1]
    lg = np.log10(ratios)
    fit = stats.linregress(lg, losses)
    xs = np.linspace(lg.min(), lg.max(), 50)
    ax.plot(10 ** xs, fit.intercept + fit.slope * xs, "-", color="#BBBBBB",
            lw=1.0, zorder=1)
    ax.plot(ratios, losses, marker="s", ms=6, color=FIXED_SCALES, ls="none",
            zorder=3)
    for n, (r, l) in enumerate(zip(ratios, losses)):
        # The leftmost label would run past the spine if it were centred.
        ha, dx = ("left", -6) if n == 0 else ("center", 0)
        ax.annotate(f"{l:+.2f}", (r, l), textcoords="offset points",
                    xytext=(dx, -13), ha=ha, fontsize=7.5, color="#555555")
    ax.axhline(0, color=COLOUR["fp32"], lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("training imbalance ratio")
    ax.set_ylabel("coverage lost to INT8 (points)")
    ax.set_xticks([1, 3, 10, 33, 100])
    ax.set_xticklabels(["1:1", "3:1", "10:1", "33:1", "100:1"])
    ax.set_ylim(min(losses) - 3.0, 2.2)
    ax.set_xlim(0.78, 128)
    ax.set_title(f"b. {fit.slope:.2f} points per decade, "
                 f"$R^2$ = {fit.rvalue ** 2:.3f}", fontsize=8.5, loc="left")

    fig.suptitle("Five seeds per ratio, threshold carried from full precision; "
                 "the effect is present and significant at 1:1", fontsize=8,
                 color="#666666", y=1.04)
    fig.tight_layout()
    return save(fig, "08_imbalance_dose_response")



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
                          ("05_temperature_scaling", fig_temperature),
                          ("06_conformal", fig_conformal),
                          ("07_scheme_decomposition",
                           fig_scheme_decomposition),
                          ("08_imbalance_dose_response",
                           fig_imbalance_dose_response)):
        builder()
        plt.close("all")
        print(" ", name)
    print("done")
