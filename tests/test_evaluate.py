"""Tests for src/evaluate.py — run with: pytest -v

These tests use small hand-calculated examples so the expected metric values
can be verified by hand, not just trusted from sklearn.
"""

import json

import pytest

from src.evaluate import compute_metrics, save_metrics


def test_perfect_predictions_give_all_ones():
    """A model that predicts everything correctly scores 1.0 everywhere."""
    metrics = compute_metrics(y_true=[0, 0, 1, 1], y_pred=[0, 0, 1, 1])
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_hand_calculated_metrics():
    """Verify metrics against values computed by hand.

    y_true = [0, 0, 1, 1, 1], y_pred = [0, 1, 1, 1, 0]
      TP = 2 (positions 2,3), FP = 1 (position 1), FN = 1 (position 4)
      accuracy    = 3/5        = 0.6
      precision   = TP/(TP+FP) = 2/3
      recall      = TP/(TP+FN) = 2/3
      f1          = 2PR/(P+R)  = 2/3
    """
    metrics = compute_metrics(y_true=[0, 0, 1, 1, 1], y_pred=[0, 1, 1, 1, 0])
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 2]]


def test_pos_label_changes_meaning():
    """With pos_label=0, precision/recall switch to describing class 0."""
    # Same predictions as above; for class 0: TP=1, FP=1, FN=1
    metrics = compute_metrics(y_true=[0, 0, 1, 1, 1], y_pred=[0, 1, 1, 1, 0], pos_label=0)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["positive_class"] == 0


def test_edge_case_never_predicting_positive():
    """zero_division=0: a model predicting NO positives yields 0.0, not NaN.

    This must stay JSON-serializable (NaN would break the JSON save).
    """
    metrics = compute_metrics(y_true=[0, 0, 1, 1, 1], y_pred=[0, 0, 0, 0, 0])
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["accuracy"] == pytest.approx(2 / 5)
    json.dumps(metrics)  # would raise if any value were NaN


def test_save_metrics_writes_valid_json(tmp_path):
    """save_metrics must create parent dirs and write parseable JSON."""
    metrics = compute_metrics(y_true=[0, 1], y_pred=[0, 1])
    out = save_metrics(metrics, tmp_path / "nested" / "metrics.json")
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["accuracy"] == 1.0
    assert loaded["n_test_samples"] == 2
