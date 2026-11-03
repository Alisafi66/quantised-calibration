"""The coverage guarantee is the whole point, so it is tested, not assumed.

Conformal prediction promises that the set contains the true label at least
(1 - alpha) of the time under exchangeability. That promise is what makes it
worth evaluating as a remedy, and an implementation that quietly breaks it
would make the remedy look like it works when it does not.
"""
from __future__ import annotations

import numpy as np
import pytest

from qcal.conformal import (aps_scores, class_conditional_coverage,
                            conformal_quantile, evaluate_sets, lac_scores,
                            mondrian_quantiles, predict_sets)


def synthetic(n=4000, n_classes=4, seed=0, skew=None):
    """Labels drawn from a model's own predictive distribution.

    Sampling the label from the predicted probabilities makes the model
    perfectly calibrated by construction, which is the setting in which the
    guarantee is easiest to state and any violation is clearly a bug.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, n_classes))
    if skew is not None:
        logits[:, 0] += skew
    e = np.exp(logits - logits.max(1, keepdims=True))
    probs = e / e.sum(1, keepdims=True)
    labels = np.array([rng.choice(n_classes, p=p) for p in probs])
    return probs, labels


def split(probs, labels, n_cal=2000):
    return probs[:n_cal], labels[:n_cal], probs[n_cal:], labels[n_cal:]


# The quantile

def test_quantile_is_monotone_in_alpha():
    scores = np.linspace(0, 1, 500)
    assert (conformal_quantile(scores, 0.01) >= conformal_quantile(scores, 0.10)
            >= conformal_quantile(scores, 0.50))


def test_quantile_is_infinite_when_the_sample_is_too_small():
    # with 5 points, ceil(6 * 0.99)/5 = 1.2 > 1, so no finite threshold works
    assert conformal_quantile(np.linspace(0, 1, 5), 0.01) == float("inf")


# The coverage guarantee

@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
@pytest.mark.parametrize("score", ["lac", "aps"])
def test_coverage_meets_the_nominal_level(alpha, score):
    probs, labels = synthetic(seed=1)
    cal_p, cal_y, test_p, test_y = split(probs, labels)
    fn = lac_scores if score == "lac" else aps_scores
    qhat = conformal_quantile(fn(cal_p, cal_y), alpha)
    sets = predict_sets(test_p, qhat, score=score)
    coverage = evaluate_sets(sets, test_y)["coverage"]
    # finite-sample slack: the guarantee is in expectation over calibration draws
    assert coverage >= (1 - alpha) - 0.03, f"{score} covered {coverage:.3f}"


def test_tighter_alpha_gives_larger_sets():
    probs, labels = synthetic(seed=2)
    cal_p, cal_y, test_p, _ = split(probs, labels)
    sizes = []
    for alpha in (0.20, 0.10, 0.01):
        qhat = conformal_quantile(lac_scores(cal_p, cal_y), alpha)
        sizes.append(predict_sets(test_p, qhat).sum(axis=1).mean())
    assert sizes[0] < sizes[1] < sizes[2], sizes


def test_aps_never_returns_an_empty_set():
    probs, labels = synthetic(seed=3)
    cal_p, cal_y, test_p, test_y = split(probs, labels)
    qhat = conformal_quantile(aps_scores(cal_p, cal_y), 0.10)
    sets = predict_sets(test_p, qhat, score="aps")
    assert evaluate_sets(sets, test_y)["empty_rate"] == 0.0


# Class-conditional calibration

def test_mondrian_lifts_coverage_on_a_class_the_pooled_form_underserves():
    """A pooled threshold can leave one class well below nominal.

    The skew makes class 0 easy and the rest harder, so a pooled threshold
    calibrated mostly on easy examples under-covers the hard ones.
    """
    probs, labels = synthetic(n=6000, seed=4, skew=3.0)
    cal_p, cal_y, test_p, test_y = split(probs, labels, n_cal=3000)
    alpha = 0.10

    pooled_q = conformal_quantile(lac_scores(cal_p, cal_y), alpha)
    pooled = class_conditional_coverage(predict_sets(test_p, pooled_q), test_y, 4)

    mondrian_q = mondrian_quantiles(lac_scores(cal_p, cal_y), cal_y, alpha, 4)
    mondrian = class_conditional_coverage(
        predict_sets(test_p, mondrian_q, score="lac"), test_y, 4)

    pooled_spread = max(pooled.values()) - min(pooled.values())
    mondrian_spread = max(mondrian.values()) - min(mondrian.values())
    assert mondrian_spread < pooled_spread, (pooled, mondrian)
    assert min(mondrian.values()) >= (1 - alpha) - 0.04, mondrian


# Reporting

def test_evaluate_sets_reports_what_it_claims():
    sets = np.array([[1, 0, 0, 0],      # singleton, correct
                     [1, 1, 0, 0],      # pair, contains truth
                     [0, 0, 0, 0],      # empty
                     [1, 0, 0, 0]],     # singleton, wrong
                    dtype=bool)
    labels = np.array([0, 1, 2, 3])
    out = evaluate_sets(sets, labels)
    assert out["coverage"] == pytest.approx(0.5)
    assert out["avg_set_size"] == pytest.approx(1.0)
    assert out["singleton_rate"] == pytest.approx(0.5)
    assert out["empty_rate"] == pytest.approx(0.25)
    assert out["singleton_precision"] == pytest.approx(0.5)
