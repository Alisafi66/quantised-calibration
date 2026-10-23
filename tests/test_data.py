"""Invariants the splits must hold, above all that nothing leaks.

The calibration split exists to give temperature scaling and conformal
prediction honest held-out data.  If a single row of it also appears in
training, every remedy RQ3 evaluates looks better than it is, and the error is
invisible in the results.  It is therefore tested rather than assumed.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from qcal.data import (
    long_tail_indices, ptq_calibration_indices, read_jsonl,
    sha256_file, stratified_holdout, write_jsonl,
)

PROC = pathlib.Path("data/processed")


@pytest.fixture
def labels():
    # four classes, 100 of each, in a deliberately non-contiguous order
    y = np.tile(np.arange(4), 100)
    return y


def test_holdout_partitions_exactly(labels):
    kept, held = stratified_holdout([""] * len(labels), labels, 0.2, seed=0)
    assert len(np.intersect1d(kept, held)) == 0
    assert len(kept) + len(held) == len(labels)


def test_holdout_keeps_class_balance(labels):
    _, held = stratified_holdout([""] * len(labels), labels, 0.25, seed=0)
    counts = np.bincount(labels[held], minlength=4)
    assert set(counts) == {25}, counts


def test_holdout_is_deterministic(labels):
    a, _ = stratified_holdout([""] * len(labels), labels, 0.2, seed=7)
    b, _ = stratified_holdout([""] * len(labels), labels, 0.2, seed=7)
    assert np.array_equal(a, b)


def test_long_tail_respects_keep_fractions(labels):
    idx = long_tail_indices(labels, [1.0, 0.3, 0.1, 0.03], seed=0)
    counts = np.bincount(labels[idx], minlength=4)
    assert list(counts) == [100, 30, 10, 3]


def test_long_tail_rejects_wrong_arity(labels):
    with pytest.raises(ValueError):
        long_tail_indices(labels, [1.0, 0.5], seed=0)


def test_ptq_sample_is_balanced_and_bounded(labels):
    idx = ptq_calibration_indices(labels, 40, seed=0)
    assert len(idx) == 40
    assert set(np.bincount(labels[idx], minlength=4)) == {10}


def test_jsonl_roundtrip_and_stable_hash(tmp_path):
    text = ["a story", "another — with punctuation", "third"]
    label = np.array([0, 2, 1])
    h1 = write_jsonl(tmp_path / "a.jsonl", text, label)
    h2 = write_jsonl(tmp_path / "b.jsonl", text, label)
    assert h1 == h2, "same content must hash identically"
    back = read_jsonl(tmp_path / "a.jsonl", "a")
    assert back.text == text
    assert np.array_equal(back.label, label)


# --- guards on the real artefacts, skipped until the snapshot is pinned -----

needs_snapshot = pytest.mark.skipif(
    not (PROC / "manifest.json").exists(),
    reason="run scripts/fetch_data.py first")


@needs_snapshot
def test_calibration_split_does_not_leak_into_training():
    train = set(read_jsonl(PROC / "train_balanced.jsonl", "t").text)
    calib = read_jsonl(PROC / "prob_calib.jsonl", "c").text
    overlap = [t for t in calib if t in train]
    assert not overlap, f"{len(overlap)} calibration rows also appear in training"


@needs_snapshot
def test_test_split_does_not_leak_into_training_or_calibration():
    seen = set(read_jsonl(PROC / "train_balanced.jsonl", "t").text)
    seen |= set(read_jsonl(PROC / "prob_calib.jsonl", "c").text)
    test = read_jsonl(PROC / "test.jsonl", "x").text
    overlap = [t for t in test if t in seen]
    assert not overlap, f"{len(overlap)} test rows appear in train or calibration"


@needs_snapshot
def test_long_tail_is_a_subset_of_training():
    train = set(read_jsonl(PROC / "train_balanced.jsonl", "t").text)
    lt = read_jsonl(PROC / "train_long_tailed.jsonl", "lt").text
    assert set(lt) <= train, "long-tailed split contains rows not in training"


@needs_snapshot
def test_manifest_hashes_match_the_files_on_disk():
    manifest = json.loads((PROC / "manifest.json").read_text())
    for name, meta in manifest["splits"].items():
        assert sha256_file(PROC / f"{name}.jsonl") == meta["sha256"], name
