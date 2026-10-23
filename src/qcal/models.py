"""The TF-IDF linear baseline.

The models in this study are instruments, not contributions.  This one earns
its place by being fast, fully deterministic and runnable on a laptop, which
makes it the right vehicle for proving the pipeline closes before any of it
reaches the GPU.  It also gives RQ1(c) its contrast: a linear model and a
transformer should not degrade the same way under quantisation, and a claim
about model capacity needs both ends of the range.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .quantise import quantise_linear_model


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TfidfLinear:
    """TF-IDF features into multinomial logistic regression."""

    def __init__(self, seed: int = 0, max_features: int = 50_000,
                 ngram_range: tuple[int, int] = (1, 2), C: float = 4.0):
        self.seed = seed
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                                   sublinear_tf=True, strip_accents="unicode",
                                   lowercase=True, min_df=2)
        self.clf = LogisticRegression(C=C, max_iter=1000, random_state=seed)

    def fit(self, text: list[str], label: np.ndarray) -> "TfidfLinear":
        X = self.vec.fit_transform(text)
        self.clf.fit(X, label)
        return self

    def features(self, text: list[str]) -> np.ndarray:
        return np.asarray(self.vec.transform(text).todense(), dtype=float)

    def predict_proba(self, text: list[str], precision: str = "fp32",
                      activation_sample: np.ndarray | None = None) -> np.ndarray:
        """Class probabilities at a given numeric precision.

        FP32 is the control.  At INT8 and INT4 both the weights and the feature
        vector pass through the integer grid first, so the logits carry the
        rounding error a quantised kernel would produce.
        """
        X = self.features(text)
        coef, intercept, quantise_activations = quantise_linear_model(
            self.clf.coef_, self.clf.intercept_, precision, activation_sample)
        return softmax(quantise_activations(X) @ coef.T + intercept)
