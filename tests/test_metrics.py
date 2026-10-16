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
