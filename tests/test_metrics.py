import pytest
import numpy as np
from qcal.metrics import (expected_calibration_error, coverage_and_precision,
                          risk_coverage_curve, aurc)


def test_perfect_calibration_has_zero_ece():
    conf = np.full(1000, 1.0)
    correct = np.ones(1000)
    assert expected_calibration_error(conf, correct) < 1e-9


def test_overconfident_model_has_positive_ece():
    conf = np.full(1000, 0.99)
    correct = np.zeros(1000)          # always wrong, always certain
    assert expected_calibration_error(conf, correct) > 0.9


def test_coverage_falls_as_threshold_rises():
    conf = np.linspace(0, 1, 100)
    correct = np.ones(100)
    low, _ = coverage_and_precision(conf, correct, 0.1)
    high, _ = coverage_and_precision(conf, correct, 0.9)
    assert low > high


def test_risk_coverage_is_monotone_in_length():
    conf = np.random.rand(500)
    correct = (np.random.rand(500) > 0.3).astype(float)
    coverage, risk = risk_coverage_curve(conf, correct)
    assert len(coverage) == len(risk) == 500
    assert 0.0 <= aurc(conf, correct) <= 1.0


def test_per_class_ece_isolates_classes():
    from qcal.metrics import per_class_ece
    # class 0 is perfectly calibrated, class 1 is confidently wrong
    conf = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    true = np.array([0, 0, 1, 1])
    out = per_class_ece(conf, correct, true, n_classes=2)
    assert out[0] == pytest.approx(0.1, abs=1e-9)
    assert out[1] == pytest.approx(0.9, abs=1e-9)


def test_bootstrap_ci_brackets_the_point_estimate():
    from qcal.metrics import bootstrap_ci, expected_calibration_error
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 500)
    correct = (rng.uniform(size=500) < conf).astype(float)
    point = expected_calibration_error(conf, correct)
    lo, hi = bootstrap_ci(expected_calibration_error, conf, correct, n_resamples=300)
    assert lo <= point <= hi
    assert hi > lo


def test_bootstrap_ci_is_reproducible():
    from qcal.metrics import bootstrap_ci, expected_calibration_error
    rng = np.random.default_rng(1)
    conf, correct = rng.uniform(0.5, 1.0, 200), rng.integers(0, 2, 200).astype(float)
    a = bootstrap_ci(expected_calibration_error, conf, correct, n_resamples=200, seed=3)
    b = bootstrap_ci(expected_calibration_error, conf, correct, n_resamples=200, seed=3)
    assert a == b


def test_threshold_maximises_coverage_subject_to_the_promise():
    from qcal.metrics import threshold_for_target_precision, coverage_and_precision
    # 90 correct at high confidence, 10 wrong at low: a 0.90 target should
    # answer nearly everything, not retreat to the very top of the ranking.
    conf = np.concatenate([np.linspace(0.99, 0.80, 90), np.linspace(0.79, 0.60, 10)])
    correct = np.concatenate([np.ones(90), np.zeros(10)])
    t = threshold_for_target_precision(conf, correct, 0.90)
    cov, prec = coverage_and_precision(conf, correct, t)
    assert prec >= 0.90
    assert cov >= 0.89, f"retreated to {cov:.2%} coverage when 90% was available"


def test_threshold_is_not_derailed_by_one_error_at_the_top():
    from qcal.metrics import threshold_for_target_precision, coverage_and_precision
    # the single most confident prediction is wrong; a naive downward scan
    # stops immediately and returns a near-useless threshold
    conf = np.linspace(0.99, 0.50, 100)
    correct = np.ones(100)
    correct[0] = 0.0
    t = threshold_for_target_precision(conf, correct, 0.95)
    cov, prec = coverage_and_precision(conf, correct, t)
    assert prec >= 0.95
    assert cov > 0.5, f"only {cov:.2%} coverage; derailed by the leading error"


def test_threshold_is_nan_when_the_promise_cannot_be_kept():
    from qcal.metrics import threshold_for_target_precision
    conf = np.linspace(0.9, 0.5, 50)
    correct = np.zeros(50)
    assert np.isnan(threshold_for_target_precision(conf, correct, 0.95))
