"""Simulated post-training quantisation for the linear baseline.

A TF-IDF classifier has no ONNX graph worth quantising, so INT8 and INT4 are
applied here as *simulated* quantisation: values are mapped onto the integer
grid a real backend would use, then mapped back, so the arithmetic downstream
sees exactly the rounding error a quantised kernel would introduce while the
code stays readable and runs anywhere.  This is the standard device for
studying quantisation effects, and it is what makes the linear baseline usable
on a laptop while the transformer waits for CUDA.

The affine scheme matches what ONNX Runtime applies to a MatMul: an asymmetric
range for activations, which are non-negative after TF-IDF, and a symmetric
per-output-channel range for weights, which straddle zero.
"""
from __future__ import annotations

import numpy as np

BITS = {"fp32": None, "int8": 8, "int4": 4}


def quant_range(bits: int, symmetric: bool) -> tuple[int, int]:
    if symmetric:
        return -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def fake_quantise(x: np.ndarray, bits: int | None, *, symmetric: bool,
                  axis: int | None = None, observed: np.ndarray | None = None) -> np.ndarray:
    """Round through an integer grid and back.

    bits     : None leaves the array untouched, which is the FP32 control
    symmetric: True for weights, False for non-negative activations
    axis     : quantise per slice along this axis (per-output-channel weights)
    observed : the range to calibrate against, when it is not x itself.  For
               activations this is the PTQ calibration sample, never the test
               set, because a deployed model cannot see its test inputs.
    """
    if bits is None:
        return x
    src = x if observed is None else observed
    qmin, qmax = quant_range(bits, symmetric)

    if axis is None:
        return _apply(x, src, qmin, qmax, symmetric)

    out = np.empty_like(x, dtype=float)
    moved_x = np.moveaxis(x, axis, 0)
    moved_src = np.moveaxis(src, axis, 0)
    moved_out = np.moveaxis(out, axis, 0)
    for i in range(moved_x.shape[0]):
        moved_out[i] = _apply(moved_x[i], moved_src[i], qmin, qmax, symmetric)
    return out


def _apply(x, src, qmin, qmax, symmetric):
    if symmetric:
        peak = float(np.abs(src).max())
        if peak == 0:
            return np.zeros_like(x, dtype=float)
        scale = peak / qmax
        zero = 0.0
    else:
        lo, hi = float(src.min()), float(src.max())
        if hi == lo:
            return np.full_like(x, lo, dtype=float)
        scale = (hi - lo) / (qmax - qmin)
        zero = qmin - lo / scale
    q = np.clip(np.round(x / scale + zero), qmin, qmax)
    return (q - zero) * scale


def quantise_linear_model(coef: np.ndarray, intercept: np.ndarray, precision: str,
                          activation_sample: np.ndarray | None = None):
    """Quantise a linear classifier the way a backend would.

    Weights go per-output-class and symmetric; the intercept is left alone,
    as backends keep biases in higher precision.  Returns the dequantised
    weights plus a function that applies the matching activation quantiser.
    """
    bits = BITS[precision]
    qcoef = fake_quantise(coef, bits, symmetric=True, axis=0)

    def quantise_activations(x: np.ndarray) -> np.ndarray:
        if bits is None or activation_sample is None:
            return x
        return fake_quantise(x, bits, symmetric=False, observed=activation_sample)

    return qcoef, intercept, quantise_activations
