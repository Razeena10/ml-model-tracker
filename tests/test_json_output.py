"""Tests for JSON output: schema shape, field values, and round-trip save/load.

Every expected value here is computed from the actual test-set predictions
(or from tiny, hand-verifiable inputs in test_evaluate.py). Nothing is
hardcoded from the README.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from src.data_loader import load_dataset
from src.evaluate import (
    CLASS_LABELS,
    CLASS_LABEL_NAMES,
    POSITIVE_CLASS_GROUPS,
    compute_metrics,
    save_metrics,
)
from train import build_pipeline, split_data


# ---------------------------------------------------------------------------
# Module-scoped fixture: one real run shared across all tests in this file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_metrics():
    """Compute metrics from a real pipeline fit on the real dataset."""
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    return compute_metrics(y_test, pipe.predict(X_test))


# ---------------------------------------------------------------------------
# Top-level schema keys
# ---------------------------------------------------------------------------


def test_top_level_keys_present(real_metrics):
    """Every documented top-level key must be present in the metrics dict."""
    required = {
        "n_test_samples",
        "positive_class_default",
        "class_names",
        "confusion_matrix",
        "metrics",
        "class_report",
        "accuracy",
        # backward-compat shim keys
        "precision",
        "recall",
        "f1",
        "positive_class",
    }
    assert required.issubset(real_metrics.keys())


def test_n_test_samples_matches_actual_split(real_metrics):
    """n_test_samples must equal the real 20% hold-out size (114)."""
    assert real_metrics["n_test_samples"] == 114


def test_accuracy_is_float_in_unit_interval(real_metrics):
    """accuracy must be a Python float in [0, 1]."""
    acc = real_metrics["accuracy"]
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# Confusion matrix sub-schema
# ---------------------------------------------------------------------------


def test_confusion_matrix_schema(real_metrics):
    """The confusion matrix must carry explicit labels, names, and orientation."""
    cm = real_metrics["confusion_matrix"]
    assert cm["labels"] == CLASS_LABELS          # [0, 1]
    assert cm["label_names"] == CLASS_LABEL_NAMES  # ["malignant", "benign"]
    assert "orientation" in cm
    matrix = cm["matrix"]
    # square, matching label count
    assert len(matrix) == len(CLASS_LABELS)
    assert all(len(row) == len(CLASS_LABELS) for row in matrix)
    # all cell values are non-negative integers
    assert all(isinstance(v, int) and v >= 0 for row in matrix for v in row)


def test_confusion_matrix_cells_sum_to_n_samples(real_metrics):
    """Cell totals must equal n_test_samples (no samples are lost/duplicated)."""
    cm = real_metrics["confusion_matrix"]
    total = sum(v for row in cm["matrix"] for v in row)
    assert total == real_metrics["n_test_samples"]


def test_confusion_matrix_true_class_rows_sum_to_class_support(real_metrics):
    """Each row total must equal the true-class count (support) for that label."""
    cm = real_metrics["confusion_matrix"]
    report = real_metrics["class_report"]
    for i, label in enumerate(CLASS_LABELS):
        row_sum = sum(cm["matrix"][i])
        support = report[str(label)]["support"]
        assert row_sum == support, (
            f"Row {i} (label {label}) sums to {row_sum} but support is {support}"
        )


# ---------------------------------------------------------------------------
# Named metric groups (benign_positive / malignant_positive)
# ---------------------------------------------------------------------------


def test_both_metric_groups_present_and_named(real_metrics):
    """Both group keys must be present with the right positive_class_name."""
    groups = real_metrics["metrics"]
    assert set(groups) == set(POSITIVE_CLASS_GROUPS)
    assert groups["benign_positive"]["positive_class"] == 1
    assert groups["benign_positive"]["positive_class_name"] == "benign"
    assert groups["malignant_positive"]["positive_class"] == 0
    assert groups["malignant_positive"]["positive_class_name"] == "malignant"


def test_metric_group_values_are_floats_in_unit_interval(real_metrics):
    """precision/recall/F1 in both groups must be floats in [0, 1]."""
    for grp in real_metrics["metrics"].values():
        for key in ("precision", "recall", "f1"):
            val = grp[key]
            assert isinstance(val, float), f"{key} is {type(val).__name__}, expected float"
            assert 0.0 <= val <= 1.0, f"{key} = {val} is out of [0, 1]"


def test_benign_group_matches_top_level_shim(real_metrics):
    """Top-level precision/recall/F1 (pos_label=1 default) must mirror benign_positive."""
    ben = real_metrics["metrics"]["benign_positive"]
    assert real_metrics["precision"] == pytest.approx(ben["precision"])
    assert real_metrics["recall"] == pytest.approx(ben["recall"])
    assert real_metrics["f1"] == pytest.approx(ben["f1"])


def test_the_two_groups_are_not_equal_on_real_data(real_metrics):
    """With the real pipeline the two groups must differ (the model is not trivial)."""
    ben = real_metrics["metrics"]["benign_positive"]
    mal = real_metrics["metrics"]["malignant_positive"]
    # Benign and malignant precision/recall are expected to differ; if they are
    # identical it indicates the dict is aliased or computed wrong.
    assert ben["precision"] != mal["precision"] or ben["recall"] != mal["recall"], (
        "benign_positive and malignant_positive are identical — check computation"
    )


# ---------------------------------------------------------------------------
# Per-class report
# ---------------------------------------------------------------------------


def test_class_report_keys_and_structure(real_metrics):
    """class_report must have one entry per label, each with the right fields."""
    report = real_metrics["class_report"]
    assert set(report.keys()) == {str(l) for l in CLASS_LABELS}
    required_fields = {"label", "name", "precision", "recall", "f1", "support"}
    for key, entry in report.items():
        assert required_fields.issubset(entry.keys()), (
            f"class_report[{key!r}] is missing fields"
        )


def test_class_report_support_sums_to_n_samples(real_metrics):
    """Sum of per-class support values must equal n_test_samples."""
    report = real_metrics["class_report"]
    total = sum(entry["support"] for entry in report.values())
    assert total == real_metrics["n_test_samples"]


def test_class_report_names_match_target_names(real_metrics):
    """Readable names in class_report must match the dataset's TARGET_NAMES."""
    from src.data_loader import TARGET_NAMES

    report = real_metrics["class_report"]
    for label in CLASS_LABELS:
        assert report[str(label)]["name"] == TARGET_NAMES[label]
        assert report[str(label)]["label"] == label


# ---------------------------------------------------------------------------
# JSON serialisation and round-trip save / load
# ---------------------------------------------------------------------------


def test_entire_metrics_dict_is_json_serialisable(real_metrics):
    """json.dumps must not raise — no numpy types, no NaN, no Inf."""
    text = json.dumps(real_metrics)  # raises on bad types
    loaded = json.loads(text)
    assert loaded["n_test_samples"] == real_metrics["n_test_samples"]


def test_save_metrics_creates_file_with_correct_schema(tmp_path, real_metrics):
    """save_metrics must write parseable JSON with all expected keys."""
    out: pathlib.Path = save_metrics(real_metrics, tmp_path / "metrics.json")
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["accuracy"] == pytest.approx(real_metrics["accuracy"])
    assert loaded["n_test_samples"] == 114
    assert set(loaded["metrics"]) == {"benign_positive", "malignant_positive"}
    assert loaded["confusion_matrix"]["matrix"] is not None
    assert loaded["confusion_matrix"]["label_names"] == ["malignant", "benign"]


def test_save_metrics_creates_parent_directories(tmp_path, real_metrics):
    """save_metrics must create intermediate directories if they don't exist."""
    nested = tmp_path / "a" / "b" / "c" / "metrics.json"
    out = save_metrics(real_metrics, nested)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["accuracy"] == pytest.approx(real_metrics["accuracy"])


def test_json_values_are_plain_python_types(real_metrics):
    """Values in the serialised JSON must be plain Python, not numpy scalars."""
    text = json.dumps(real_metrics)
    loaded = json.loads(text)
    # accuracy, precision etc. must deserialise as float (not numpy.float64)
    assert type(loaded["accuracy"]) is float
    assert type(loaded["n_test_samples"]) is int
    for grp in loaded["metrics"].values():
        assert type(grp["precision"]) is float
        assert type(grp["recall"]) is float
        assert type(grp["f1"]) is float
