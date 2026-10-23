"""Calibration and selective-prediction metrics.

Implemented here rather than imported so that the binning scheme and the
definition of "delivered precision" are explicit and testable — both are
reported in the dissertation and both are sensitive to how they are computed.
"""
from __future__ import annotations
import numpy as np


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width binning ECE (Guo et al., 2017).

    conf    : predicted probability of the predicted class, shape (n,)
    correct : 1 where the prediction was right, shape (n,)
    """
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def coverage_and_precision(conf: np.ndarray, correct: np.ndarray, threshold: float):
    """What the model actually does at a given abstention threshold.

    Returns (coverage, delivered_precision). Coverage is the fraction of inputs
    answered; delivered precision is accuracy among those answered.
    """
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    answered = conf >= threshold
    coverage = float(answered.mean())
    delivered = float(correct[answered].mean()) if answered.any() else float("nan")
    return coverage, delivered


def threshold_for_target_precision(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Lowest threshold whose delivered precision still meets the target.

    This is the threshold a practitioner would set at full precision — and the
    one RQ2 carries across to the quantised model unchanged.
    """
    for t in np.unique(np.asarray(conf, float))[::-1]:
        _, delivered = coverage_and_precision(conf, correct, t)
        if not np.isnan(delivered) and delivered >= target:
            best = t
        else:
            break
    return float(locals().get("best", 1.0))


def risk_coverage_curve(conf: np.ndarray, correct: np.ndarray):
    """Selective risk against coverage (El-Yaniv & Wiener, 2010)."""
    order = np.argsort(-np.asarray(conf, float))
    c = np.asarray(correct, float)[order]
    k = np.arange(1, len(c) + 1)
    coverage = k / len(c)
    risk = np.cumsum(1.0 - c) / k
    return coverage, risk


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk–coverage curve. Lower is better."""
    coverage, risk = risk_coverage_curve(conf, correct)
    return float(np.trapezoid(risk, coverage))


def per_class_ece(conf: np.ndarray, correct: np.ndarray, true_label: np.ndarray,
                  n_classes: int, n_bins: int = 15) -> dict[int, float]:
    """ECE computed separately within each true class.

    RQ1(b) asks whether calibration damage is uniform or concentrates in the
    classes a model saw rarely, which a single pooled figure cannot answer: a
    large error on a small class is invisible once it is averaged against a
    well-calibrated majority.
    """
    true_label = np.asarray(true_label)
    out: dict[int, float] = {}
    for c in range(n_classes):
        m = true_label == c
        out[c] = (expected_calibration_error(conf[m], correct[m], n_bins)
                  if m.sum() else float("nan"))
    return out


def bootstrap_ci(statistic, *arrays: np.ndarray, n_resamples: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval over the evaluation set.

    A deterministic model fitted on fixed data gives the same answer at every
    seed, so seed-to-seed spread cannot express the uncertainty in a figure
    like ECE.  Resampling the test set can: it asks how far the number would
    move had the evaluation drawn a different sample of the same size, which
    is the question a reader of the results actually has.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    vals = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        vals[b] = statistic(*(a[idx] for a in arrays))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
