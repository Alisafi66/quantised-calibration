"""AG News loading, splitting and the long-tailed variant.

Three decisions are made here and each is reported in the methodology.

The *probability* calibration split is carved out of AG News's training set and
kept disjoint from everything the model fits on.  Temperature scaling and split
conformal prediction both need held-out data, and reusing training data for it
would flatter every remedy RQ3 tests.

The *quantisation* calibration sample, the batch a PTQ backend uses to choose
scaling factors, is drawn from the training portion instead.  The two are
different things that share a name, and keeping them apart is the difference
between a clean result and a leak.

The long-tailed variant imbalances only the training split.  The question in
RQ1(b) is whether calibration damage concentrates in classes the model saw
rarely, so the imbalance belongs in what the model learns from; the test set
stays balanced, because measuring per-class calibration needs enough examples
of every class to measure.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass

import numpy as np

AG_NEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass(frozen=True)
class Split:
    """One partition: raw text, integer labels, and where it came from."""
    text: list[str]
    label: np.ndarray
    name: str

    def __len__(self) -> int:
        return len(self.text)

    def class_counts(self) -> dict[str, int]:
        return {AG_NEWS_CLASSES[i]: int((self.label == i).sum())
                for i in range(len(AG_NEWS_CLASSES))}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: pathlib.Path, text: list[str], label: np.ndarray) -> str:
    """Serialise a split deterministically and return its hash.

    The hash is taken over this file rather than over whatever the download
    cache happens to hold, because only a fixed field order and a fixed row
    order produce the same digest twice.

    `newline="\n"` is not cosmetic.  Python's text mode translates newlines to
    the platform's convention, so the same rows written on Windows and on macOS
    differ by one byte per line and hash differently while being identical in
    content.  Pinning the terminator is what makes the digest a statement about
    the data rather than about the machine that wrote it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for t, y in zip(text, label):
            f.write(json.dumps({"text": t, "label": int(y)},
                               ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(path)


def read_jsonl(path: pathlib.Path, name: str) -> Split:
    text, label = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text.append(row["text"])
            label.append(row["label"])
    return Split(text, np.asarray(label, dtype=np.int64), name)


def stratified_holdout(text: list[str], label: np.ndarray, fraction: float, seed: int):
    """Split off `fraction` of each class, so both sides keep the same mix.

    Returns (kept, held_out) as index arrays.
    """
    rng = np.random.default_rng(seed)
    held: list[np.ndarray] = []
    for c in np.unique(label):
        idx = np.flatnonzero(label == c)
        rng.shuffle(idx)
        held.append(idx[: int(round(fraction * len(idx)))])
    held_idx = np.sort(np.concatenate(held))
    mask = np.ones(len(label), dtype=bool)
    mask[held_idx] = False
    return np.flatnonzero(mask), held_idx


def long_tail_indices(label: np.ndarray, keep_fractions: list[float], seed: int) -> np.ndarray:
    """Induce a controlled class imbalance by keeping a fraction of each class.

    AG News is balanced by construction, which leaves RQ1(b) with no rare class
    to look at.  Rather than adopt a second corpus, the tail is built from this
    one: classes are ordered by their label index and the i-th class keeps
    `keep_fractions[i]` of its examples, so the imbalance is a property of the
    design rather than of the data source, and its exact shape is recorded.
    """
    classes = np.unique(label)
    if len(keep_fractions) != len(classes):
        raise ValueError(
            f"{len(keep_fractions)} keep fractions for {len(classes)} classes")
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    for c, frac in zip(classes, keep_fractions):
        idx = np.flatnonzero(label == c)
        rng.shuffle(idx)
        kept.append(idx[: int(round(frac * len(idx)))])
    return np.sort(np.concatenate(kept))


def ptq_calibration_indices(label: np.ndarray, n: int, seed: int) -> np.ndarray:
    """A class-balanced sample for the quantisation backend to calibrate on.

    This is the PTQ calibration set, not the probability-calibration split.  It
    is drawn from training data because that is what a practitioner would have
    at quantisation time, and it is balanced so that scaling factors are not
    chosen from a skewed view of the activations.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(label)
    per_class = max(1, n // len(classes))
    picked: list[np.ndarray] = []
    for c in classes:
        idx = np.flatnonzero(label == c)
        rng.shuffle(idx)
        picked.append(idx[:per_class])
    return np.sort(np.concatenate(picked))[:n]
