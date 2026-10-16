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
