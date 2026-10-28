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

# ---------------------------------------------------------------------------
# Split behavior: sizes, stratification, disjointness, and a full repeat cycle
# ---------------------------------------------------------------------------

from train import TEST_SIZE  # noqa: E402  (import after module docstring)


def test_split_sizes_match_test_size():
    """A 20% hold-out of 569 samples must give 455 train / 114 test."""
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == 455
    assert len(X_test) == 114
    assert len(y_train) == 455
    assert len(y_test) == 114
    # sanity: the fractions add up to the whole dataset
    assert len(X_train) + len(X_test) == len(X)


def test_split_preserves_class_proportions():
    """Stratification must keep the malignant/benign ratio in both halves."""
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    for label in (0, 1):
        total = int((y == label).sum())
        in_train = int((y_train == label).sum())
        in_test = int((y_test == label).sum())
        # no samples lost
        assert in_train + in_test == total
        # the class ratio in each half matches the overall ratio (within ~1)
        overall_ratio = total / len(y)
        assert abs(in_train / len(y_train) - overall_ratio) < 0.01
        assert abs(in_test / len(y_test) - overall_ratio) < 0.01


def test_train_and_test_indices_are_disjoint():
    """No sample may appear in both the training and the test set."""
    X, y = load_dataset()
    X_train, X_test, _, _ = split_data(X, y)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_repeat_full_training_cycle_gives_identical_metrics():
    """Two independent load -> split -> fit -> evaluate cycles must match.

    Unlike test_repeated_training_gives_identical_metrics (which reuses one
    split), this re-runs the entire pipeline from data loading, exercising the
    reproducibility of the seed end-to-end.
    """
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    m1 = compute_metrics(y_test, build_pipeline().fit(X_train, y_train).predict(X_test))

    X2, y2 = load_dataset()
    X_train2, X_test2, y_train2, y_test2 = split_data(X2, y2)
    m2 = compute_metrics(y_test2, build_pipeline().fit(X_train2, y_train2).predict(X_test2))

    for key in ("accuracy", "precision", "recall", "f1"):
        assert m1[key] == m2[key], f"{key} changed between independent runs"
    assert m1["confusion_matrix"] == m2["confusion_matrix"]
