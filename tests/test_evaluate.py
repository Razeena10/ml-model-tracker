"""Tests for src/evaluate.py — run with: pytest -v

These tests use small hand-calculated examples so the expected metric values
can be verified by hand, not just trusted from sklearn. Every expected number
below is computed from the test inputs on the line above it; none are copied
from the README or from a real training run.

The evaluator now reports TWO positive-class groups side by side:
  * benign_positive   (positive class = 1 = benign)
  * malignant_positive (positive class = 0 = malignant)
plus a per-class report and an explicitly-labeled confusion matrix whose order
is always [malignant, benign] (labels [0, 1]).
"""

import json

import pytest

from src.evaluate import compute_metrics, save_metrics


# ---------------------------------------------------------------------------
# Fixtures of tiny, hand-verifiable inputs
# ---------------------------------------------------------------------------

# Perfect predictions: everything correct.
PERFECT = ([0, 0, 1, 1], [0, 0, 1, 1])

# Mixed predictions for hand-calculation.
# y_true = [0, 0, 1, 1, 1], y_pred = [0, 1, 1, 1, 0]
#   confusion matrix (rows=true, cols=pred, order [0,1]): [[1, 1], [1, 2]]
#   accuracy = 3/5 = 0.6
#   benign (1): TP=2, FP=1, FN=1 -> P=R=F1=2/3
#   malignant (0): TP=1, FP=1, FN=1 -> P=R=F1=1/2=0.5
MIXED = ([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])

# Never predicts benign (predicts only 0).
# y_true = [0, 0, 1, 1, 1], y_pred = [0, 0, 0, 0, 0]
#   confusion matrix: [[2, 0], [3, 0]]
#   accuracy = 2/5 = 0.4
#   benign (1): never predicted -> P=R=F1=0 (zero_division=0)
#   malignant (0): TP=2, FP=3, FN=0 -> P=2/5=0.4, R=1.0, F1=4/7
NEVER_BENIGN = ([0, 0, 1, 1, 1], [0, 0, 0, 0, 0])

# Never predicts malignant (predicts only 1).
# y_true = [0, 0, 1, 1, 1], y_pred = [1, 1, 1, 1, 1]
#   confusion matrix: [[0, 2], [0, 3]]
#   accuracy = 3/5 = 0.6
#   benign (1): TP=3, FP=2, FN=0 -> P=3/5=0.6, R=1.0, F1=3/4=0.75
#   malignant (0): never predicted -> P=R=F1=0 (zero_division=0)
NEVER_MALIGNANT = ([0, 0, 1, 1, 1], [1, 1, 1, 1, 1])


# ---------------------------------------------------------------------------
# Existing behavior preserved (accuracy + both groups on the simple cases)
# ---------------------------------------------------------------------------


def test_perfect_predictions_give_all_ones():
    """A model that predicts everything correctly scores 1.0 everywhere."""
    metrics = compute_metrics(y_true=PERFECT[0], y_pred=PERFECT[1])
    # backward-compatible top-level keys
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    # confusion matrix now carries explicit labels and order
    cm = metrics["confusion_matrix"]
    assert cm["matrix"] == [[2, 0], [0, 2]]
    assert cm["labels"] == [0, 1]
    # both groups must be perfect too
    assert metrics["metrics"]["benign_positive"]["precision"] == 1.0
    assert metrics["metrics"]["malignant_positive"]["recall"] == 1.0


def test_hand_calculated_metrics():
    """Verify metrics against values computed by hand (see MIXED above)."""
    metrics = compute_metrics(y_true=MIXED[0], y_pred=MIXED[1])
    assert metrics["accuracy"] == pytest.approx(0.6)
    # top-level defaults to benign-positive (POSITIVE_CLASS = 1)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    # confusion matrix with explicit order
    assert metrics["confusion_matrix"]["matrix"] == [[1, 1], [1, 2]]
    assert metrics["confusion_matrix"]["label_names"] == ["malignant", "benign"]


# ---------------------------------------------------------------------------
# Both positive-class definitions
# ---------------------------------------------------------------------------


def test_both_positive_class_groups_present_and_named():
    """compute_metrics must always return both named groups with clear names."""
    metrics = compute_metrics(y_true=MIXED[0], y_pred=MIXED[1])
    groups = metrics["metrics"]
    assert set(groups) == {"benign_positive", "malignant_positive"}

    # benign-positive: P=R=F1=2/3 (TP=2, FP=1, FN=1)
    ben = groups["benign_positive"]
    assert ben["positive_class"] == 1
    assert ben["positive_class_name"] == "benign"
    assert ben["precision"] == pytest.approx(2 / 3)
    assert ben["recall"] == pytest.approx(2 / 3)
    assert ben["f1"] == pytest.approx(2 / 3)

    # malignant-positive: P=R=F1=0.5 (TP=1, FP=1, FN=1)
    mal = groups["malignant_positive"]
    assert mal["positive_class"] == 0
    assert mal["positive_class_name"] == "malignant"
    assert mal["precision"] == pytest.approx(0.5)
    assert mal["recall"] == pytest.approx(0.5)
    assert mal["f1"] == pytest.approx(0.5)


def test_pos_label_switches_top_level_but_not_the_named_groups():
    """pos_label only selects the backward-compat top-level shim.

    The named groups are ALWAYS both present and unchanged by pos_label, so
    callers can never accidentally read malignant numbers as benign ones.
    """
    metrics = compute_metrics(y_true=MIXED[0], y_pred=MIXED[1], pos_label=0)
    # top-level shim now reflects malignant-positive
    assert metrics["positive_class"] == 0
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    # ...but the named groups are untouched and still both present
    assert metrics["metrics"]["benign_positive"]["precision"] == pytest.approx(2 / 3)
    assert metrics["metrics"]["malignant_positive"]["precision"] == pytest.approx(0.5)


def test_groups_are_distinct_when_classes_are_not_symmetric():
    """The two groups must diverge when errors are unevenly distributed.

    With NEVER_BENIGN the model catches every malignancy (malignant recall=1.0)
    but misses every benign case (benign recall=0.0). A single positive class
    would hide this; reporting both exposes it.
    """
    metrics = compute_metrics(y_true=NEVER_BENIGN[0], y_pred=NEVER_BENIGN[1])
    assert metrics["metrics"]["benign_positive"]["recall"] == 0.0
    assert metrics["metrics"]["malignant_positive"]["recall"] == 1.0


# ---------------------------------------------------------------------------
# Confusion matrix ordering and explicit labels
# ---------------------------------------------------------------------------


def test_confusion_matrix_ordering_and_labels():
    """Confusion matrix rows/cols are always [malignant, benign] (labels [0,1]).

    Using MIXED: true 0s -> [1 predicted 0, 1 predicted 1]; true 1s ->
    [1 predicted 0, 2 predicted 1] => [[1, 1], [1, 2]].
    """
    metrics = compute_metrics(y_true=MIXED[0], y_pred=MIXED[1])
    cm = metrics["confusion_matrix"]
    assert cm["labels"] == [0, 1]
    assert cm["label_names"] == ["malignant", "benign"]
    assert cm["matrix"] == [[1, 1], [1, 2]]
    assert cm["orientation"] == "rows=true_class, cols=predicted_class"
    # rows correspond to true labels in the same order as label_names
    assert len(cm["matrix"]) == len(cm["label_names"])
    assert all(len(row) == len(cm["label_names"]) for row in cm["matrix"])


# ---------------------------------------------------------------------------
# Per-class report
# ---------------------------------------------------------------------------


def test_class_report_has_named_classes_and_supports():
    """The class report carries explicit labels, readable names, and supports."""
    metrics = compute_metrics(y_true=MIXED[0], y_pred=MIXED[1])
    report = metrics["class_report"]
    assert set(report) == {"0", "1"}
    # class 0 (malignant): P=R=F1=0.5, support=2 (two true 0s)
    assert report["0"]["label"] == 0
    assert report["0"]["name"] == "malignant"
    assert report["0"]["precision"] == pytest.approx(0.5)
    assert report["0"]["recall"] == pytest.approx(0.5)
    assert report["0"]["f1"] == pytest.approx(0.5)
    assert report["0"]["support"] == 2
    # class 1 (benign): P=R=F1=2/3, support=3 (three true 1s)
    assert report["1"]["name"] == "benign"
    assert report["1"]["recall"] == pytest.approx(2 / 3)
    assert report["1"]["support"] == 3
    # supports sum to the number of samples
    assert report["0"]["support"] + report["1"]["support"] == 5


# ---------------------------------------------------------------------------
# Zero-division edge cases (both directions)
# ---------------------------------------------------------------------------


def test_edge_case_never_predicting_positive_benign():
    """Never predicting benign: benign group is 0.0 (not NaN), stays JSON-safe."""
    metrics = compute_metrics(y_true=NEVER_BENIGN[0], y_pred=NEVER_BENIGN[1])
    ben = metrics["metrics"]["benign_positive"]
    assert ben["precision"] == 0.0
    assert ben["recall"] == 0.0
    assert ben["f1"] == 0.0
    assert metrics["accuracy"] == pytest.approx(2 / 5)
    # the other group must still be correct and non-NaN
    mal = metrics["metrics"]["malignant_positive"]
    assert mal["precision"] == pytest.approx(2 / 5)
    assert mal["recall"] == 1.0
    assert mal["f1"] == pytest.approx(4 / 7)
    json.dumps(metrics)  # would raise if any value were NaN


def test_edge_case_never_predicting_positive_malignant():
    """Never predicting malignant: malignant group is 0.0, benign group nonzero.

    This is the symmetric zero-division case and the one that matters most for
    a screening task: a model that flags everything as benign never misses a
    malignancy call (it makes none), so malignant precision/recall/F1 are 0.0.
    """
    metrics = compute_metrics(y_true=NEVER_MALIGNANT[0], y_pred=NEVER_MALIGNANT[1])
    mal = metrics["metrics"]["malignant_positive"]
    assert mal["precision"] == 0.0
    assert mal["recall"] == 0.0
    assert mal["f1"] == 0.0
    assert metrics["accuracy"] == pytest.approx(3 / 5)
    ben = metrics["metrics"]["benign_positive"]
    assert ben["precision"] == pytest.approx(3 / 5)
    assert ben["recall"] == 1.0
    assert ben["f1"] == pytest.approx(3 / 4)
    json.dumps(metrics)


# ---------------------------------------------------------------------------
# JSON serialization and saving
# ---------------------------------------------------------------------------


def test_full_schema_is_json_serializable():
    """The entire metrics dict must round-trip through JSON without loss."""
    metrics = compute_metrics(y_true=MIXED[0], y_pred=MIXED[1])
    text = json.dumps(metrics)  # must not raise (no NaN, no numpy types)
    loaded = json.loads(text)
    assert loaded["n_test_samples"] == 5
    assert loaded["positive_class_default"] == 1
    assert loaded["class_names"] == {"0": "malignant", "1": "benign"}
    assert set(loaded["metrics"]) == {"benign_positive", "malignant_positive"}
    assert set(loaded["class_report"]) == {"0", "1"}
    assert loaded["confusion_matrix"]["matrix"] == [[1, 1], [1, 2]]
    # accuracy is a global metric, present once at top level
    assert loaded["accuracy"] == pytest.approx(0.6)


def test_save_metrics_writes_valid_json(tmp_path):
    """save_metrics must create parent dirs and write parseable JSON."""
    metrics = compute_metrics(y_true=[0, 1], y_pred=[0, 1])
    out = save_metrics(metrics, tmp_path / "nested" / "metrics.json")
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    # backward-compatible top-level keys
    assert loaded["accuracy"] == 1.0
    assert loaded["n_test_samples"] == 2
    # new schema present
    assert loaded["confusion_matrix"]["matrix"] == [[1, 0], [0, 1]]
    assert loaded["confusion_matrix"]["label_names"] == ["malignant", "benign"]
    assert loaded["metrics"]["benign_positive"]["precision"] == 1.0
    assert loaded["metrics"]["malignant_positive"]["recall"] == 1.0
    assert loaded["class_report"]["0"]["name"] == "malignant"
    assert loaded["class_report"]["1"]["support"] == 1
