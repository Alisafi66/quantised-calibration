"""Split conformal prediction, implemented here for the same reason as the metrics.

Conformal prediction makes a distribution-free promise: if the calibration and
test data are exchangeable, the prediction set contains the true label at least
(1 - alpha) of the time, whatever the model underneath is doing. That promise
holds even for a model whose probabilities are badly wrong, which is exactly
why it is worth testing against a quantised model.

What the promise does not cover is *usefulness*. A set containing every class
is trivially valid and tells a practitioner nothing. Validity is therefore
reported alongside efficiency throughout: average set size, how often the set
is a single label, and how often it is empty.

Two score functions are provided.

LAC (also called THR) scores a label by one minus its predicted probability.
It produces the smallest sets at a given coverage, but it spends its error
budget wherever the model is weakest, so coverage within a class can fall well
below the nominal level even while the pooled figure is met.

APS accumulates sorted probability mass until the true label is reached. Sets
are larger, and conditional coverage is better behaved.

Both are offered in a Mondrian (class-conditional) form, which calibrates a
separate threshold per class. El Allam and Hamlich (2025) adopt the same
construction for an imbalanced medical task, on the argument that a minority
class warrants its own threshold; they report no pooled comparison.
"""
from __future__ import annotations

import numpy as np


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """The finite-sample corrected (1 - alpha) quantile of calibration scores.

    The correction, ceil((n + 1)(1 - alpha)) / n rather than simply 1 - alpha,
    is what makes the coverage guarantee hold at finite n rather than only
    asymptotically. With too few calibration points the required quantile can
    exceed 1, which means no finite threshold achieves the target and the set
    must contain every label; inf is returned to say so explicitly.
    """
    cal_scores = np.asarray(cal_scores, float)
    n = len(cal_scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return float("inf")
    return float(np.quantile(cal_scores, level, method="higher"))


def lac_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Nonconformity of the true label: one minus its predicted probability."""
    probs = np.asarray(probs, float)
    return 1.0 - probs[np.arange(len(labels)), np.asarray(labels, int)]


def aps_scores(probs: np.ndarray, labels: np.ndarray, *,
               randomise: bool = True, seed: int = 0) -> np.ndarray:
    """Cumulative probability mass down to the true label.

    Classes are visited in descending probability and their mass accumulated
    until the true label is included. Randomising the contribution of that last
    label removes the conservatism that otherwise makes coverage overshoot.
    """
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, int)
    order = np.argsort(-probs, axis=1)
    ordered = np.take_along_axis(probs, order, axis=1)
    cumulative = ordered.cumsum(axis=1)
    rank = np.argmax(order == labels[:, None], axis=1)
    rows = np.arange(len(labels))
    total = cumulative[rows, rank]
    if randomise:
        u = np.random.default_rng(seed).uniform(size=len(labels))
        total = total - u * ordered[rows, rank]
    return total


def _aps_set_mask(probs: np.ndarray, qhat: np.ndarray | float, *,
                  randomise: bool, seed: int) -> np.ndarray:
    probs = np.asarray(probs, float)
    order = np.argsort(-probs, axis=1)
    ordered = np.take_along_axis(probs, order, axis=1)
    cumulative = ordered.cumsum(axis=1)
    threshold = np.asarray(qhat, float)
    if threshold.ndim == 0:
        threshold = np.full(len(probs), float(threshold))
    if randomise:
        u = np.random.default_rng(seed).uniform(size=(len(probs), 1))
        include = (cumulative - u * ordered) <= threshold[:, None]
    else:
        include = (cumulative - ordered) <= threshold[:, None]
    include[:, 0] = True          # never return an empty set under APS
    mask = np.zeros_like(include)
    np.put_along_axis(mask, order, include, axis=1)
    return mask


def predict_sets(probs: np.ndarray, qhat: np.ndarray | float, *, score: str = "lac",
                 randomise: bool = True, seed: int = 0) -> np.ndarray:
    """Boolean mask of shape (n, classes): which labels the set contains.

    qhat may be a scalar, or one threshold per class for the Mondrian form.
    """
    probs = np.asarray(probs, float)
    if score == "lac":
        threshold = np.asarray(qhat, float)
        if threshold.ndim == 0:
            return (1.0 - probs) <= float(threshold)
        return (1.0 - probs) <= threshold[None, :]   # per-class thresholds
    if score == "aps":
        return _aps_set_mask(probs, qhat, randomise=randomise, seed=seed)
    raise ValueError(f"unknown score function: {score}")


def mondrian_quantiles(cal_scores: np.ndarray, cal_labels: np.ndarray,
                       alpha: float, n_classes: int) -> np.ndarray:
    """One threshold per class, calibrated only on that class's examples.

    A pooled threshold spends the error budget wherever examples are plentiful,
    so a class the model handles badly can be far below nominal coverage while
    the overall figure looks correct. Calibrating within each class prevents
    that at the cost of a smaller calibration sample per threshold.
    """
    cal_labels = np.asarray(cal_labels, int)
    return np.array([
        conformal_quantile(cal_scores[cal_labels == c], alpha)
        if (cal_labels == c).any() else float("inf")
        for c in range(n_classes)
    ])


def evaluate_sets(sets: np.ndarray, labels: np.ndarray) -> dict:
    """Validity and efficiency together, because validity alone is not useful."""
    labels = np.asarray(labels, int)
    rows = np.arange(len(labels))
    sizes = sets.sum(axis=1)
    covered = sets[rows, labels]
    singleton = sizes == 1
    return {
        "coverage": float(covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "singleton_rate": float(singleton.mean()),
        "empty_rate": float((sizes == 0).mean()),
        # accuracy among the inputs the method is willing to answer outright,
        # which is the quantity comparable to a thresholded abstention rule
        "singleton_precision": float(covered[singleton].mean()) if singleton.any()
                               else float("nan"),
    }


def class_conditional_coverage(sets: np.ndarray, labels: np.ndarray,
                               n_classes: int) -> dict[int, float]:
    """Coverage within each true class, which a pooled figure can conceal."""
    labels = np.asarray(labels, int)
    rows = np.arange(len(labels))
    covered = sets[rows, labels]
    return {c: float(covered[labels == c].mean()) if (labels == c).any() else float("nan")
            for c in range(n_classes)}
