"""Tests for reproducibility guarantees (Step 6, requirement 6).

If these ever fail, results in README/MLflow can no longer be reproduced —
treat that as a bug, not flakiness.
"""

import numpy as np

from src.data_loader import load_dataset
from src.evaluate import compute_metrics
from train import build_pipeline, split_data


def test_same_seed_gives_identical_splits():
    """random_state=42 must yield the exact same 4 arrays every time."""
    X, y = load_dataset()
    a_train, a_test, ay_train, ay_test = split_data(X, y)
    b_train, b_test, by_train, by_test = split_data(X, y)
    assert (a_train.to_numpy() == b_train.to_numpy()).all()
    assert (a_test.to_numpy() == b_test.to_numpy()).all()
    assert (ay_train.to_numpy() == by_train.to_numpy()).all()
    assert (ay_test.to_numpy() == by_test.to_numpy()).all()


def test_repeated_training_gives_identical_metrics():
    """Two full fits on the same split must produce identical metrics.

    LogisticRegression on this data is deterministic (same seed, same
    solver path), so exact equality is expected here — not a tolerance.
    """
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    m1 = compute_metrics(y_test, build_pipeline().fit(X_train, y_train).predict(X_test))
    m2 = compute_metrics(y_test, build_pipeline().fit(X_train, y_train).predict(X_test))

    for key in ("accuracy", "precision", "recall", "f1"):
        assert m1[key] == m2[key], f"{key} changed between identical runs"
    assert m1["confusion_matrix"] == m2["confusion_matrix"]
