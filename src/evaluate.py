"""Metric calculation for the classification pipeline (Step 4).

This module reports classification metrics for the Breast Cancer Wisconsin
binary classifier, where the labels are:

    0 = malignant  (cancerous)
    1 = benign     (not cancerous)

Two positive-class definitions are reported side by side so they can never be
confused:

* **benign-positive** (positive class = 1, scikit-learn's default for this
  dataset): precision/recall/F1 describe how well the model detects *benign*
  cases.
* **malignant-positive** (positive class = 0): the same measurements
  reinterpreted for *malignant* detection — the clinically important error to
  avoid, since a missed malignancy is far more dangerous than a false alarm.

It also reports per-class precision/recall/F1/support and a confusion matrix
whose row/column order is explicit: ``[malignant, benign]`` (labels ``[0, 1]``).

NOTE — this is an evaluation module, not a clinical validation. The metrics
here measure agreement with a held-out test set for a teaching project. They
do NOT certify that the model is medically safe or suitable for clinical use.

Backward compatibility
---------------------
The pre-existing callers (``train.py`` and ``src/tracking.log_metrics_dict``)
read top-level ``accuracy``/``precision``/``recall``/``f1``. Those keys are kept
at the top level. The top-level precision/recall/F1 follow the ``pos_label``
argument (default ``POSITIVE_CLASS`` = 1 = benign), matching the prior
behavior. The new, clearly-named groups under ``metrics`` are the canonical,
unambiguous source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from src.data_loader import POSITIVE_CLASS, TARGET_NAMES

# Fixed label order used everywhere in this module: [malignant, benign].
# Keeping this constant makes the confusion-matrix orientation unambiguous.
CLASS_LABELS = [0, 1]
CLASS_LABEL_NAMES = [TARGET_NAMES[label] for label in CLASS_LABELS]  # ["malignant", "benign"]

# The two positive-class groups reported in every metrics dict. Keying them
# with descriptive names (not just integers) prevents anyone from confusing
# benign-detection numbers with malignant-detection numbers.
POSITIVE_CLASS_GROUPS = {
    "benign_positive": 1,
    "malignant_positive": 0,
}


def _group_metrics(y_true, y_pred, positive_class: int) -> dict:
    """Precision/recall/F1 for one chosen positive class (zero_division=0)."""
    return {
        "positive_class": int(positive_class),
        "positive_class_name": TARGET_NAMES[int(positive_class)],
        "precision": float(
            precision_score(y_true, y_pred, pos_label=positive_class, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=positive_class, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_class, zero_division=0)),
    }


def compute_metrics(y_true, y_pred, pos_label: int = POSITIVE_CLASS) -> dict:
    """Compute evaluation metrics for a binary classifier.

    The returned dict always contains BOTH positive-class groups
    (``metrics.benign_positive`` and ``metrics.malignant_positive``), a
    per-class ``class_report``, and an explicitly-labeled ``confusion_matrix``.

    For backward compatibility, top-level ``accuracy``/``precision``/``recall``/
    ``f1`` are also present. The top-level precision/recall/F1 follow the
    ``pos_label`` argument (default ``POSITIVE_CLASS`` = 1 = benign), matching
    the pre-existing behavior and the values logged to MLflow by
    ``src.tracking.log_metrics_dict``.

    ``zero_division=0`` is used throughout: a degenerate model that never
    predicts the positive class yields 0.0 (not NaN), keeping results
    JSON-serializable and honest ("we never predicted positive").
    """
    accuracy = float(accuracy_score(y_true, y_pred))

    # Confusion matrix with an explicit, fixed label order: [malignant, benign].
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    # Per-class precision/recall/F1/support, in the fixed [malignant, benign]
    # order. average=None returns one value per label.
    per_prec, per_rec, per_f1, per_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    class_report = {
        str(label): {
            "label": int(label),
            "name": TARGET_NAMES[label],
            "precision": float(per_prec[i]),
            "recall": float(per_rec[i]),
            "f1": float(per_f1[i]),
            "support": int(per_supp[i]),
        }
        for i, label in enumerate(CLASS_LABELS)
    }

    # Both named positive-class groups, always present.
    groups = {
        name: _group_metrics(y_true, y_pred, positive_class)
        for name, positive_class in POSITIVE_CLASS_GROUPS.items()
    }

    metrics = {
        "n_test_samples": int(len(y_true)),
        "positive_class_default": int(POSITIVE_CLASS),
        "class_names": {str(label): TARGET_NAMES[label] for label in CLASS_LABELS},
        "confusion_matrix": {
            "labels": list(CLASS_LABELS),
            "label_names": list(CLASS_LABEL_NAMES),
            "matrix": cm.tolist(),
            "orientation": "rows=true_class, cols=predicted_class",
        },
        "metrics": groups,
        "class_report": class_report,
        "accuracy": accuracy,
    }

    # Backward-compatible top-level precision/recall/F1 that follow `pos_label`.
    # These mirror the values in the corresponding named group above so there
    # is exactly one source of truth (the group); the top-level keys are a
    # thin shim for existing callers.
    primary = groups["benign_positive"] if pos_label == 1 else groups["malignant_positive"]
    metrics["precision"] = primary["precision"]
    metrics["recall"] = primary["recall"]
    metrics["f1"] = primary["f1"]
    metrics["positive_class"] = int(pos_label)
    return metrics


def print_metrics(metrics: dict) -> None:
    """Print the metrics dict in a human-readable, clearly-labeled format.

    Both positive-class groups are printed so benign-detection and
    malignant-detection numbers are never confused. This is an evaluation
    report, not a clinical validation (see the module docstring).
    """
    print("\nTEST SET EVALUATION")
    print("-" * 64)
    print(f"Test samples : {metrics['n_test_samples']}")
    print(f"Accuracy     : {metrics['accuracy']:.4f}  (global; same for both groups)")
    cm = metrics["confusion_matrix"]
    print(f"Class order  : {cm['label_names']} (labels {cm['labels']})")

    print("\nMetrics by positive class")
    print("-" * 64)
    for group_name, group in metrics["metrics"].items():
        print(
            f"  {group_name}  (positive class = {group['positive_class']} "
            f"= {group['positive_class_name']}):"
        )
        print(f"    precision : {group['precision']:.4f}")
        print(f"    recall    : {group['recall']:.4f}")
        print(f"    f1        : {group['f1']:.4f}")

    print("\nPer-class report (precision/recall/F1 for each true class)")
    print("-" * 64)
    print(
        f"  {'label':<6}{'name':<12}{'precision':<12}{'recall':<12}{'f1':<12}{'support':<8}"
    )
    for label_str, row in metrics["class_report"].items():
        print(
            f"  {label_str:<6}{row['name']:<12}{row['precision']:<12.4f}"
            f"{row['recall']:<12.4f}{row['f1']:<12.4f}{row['support']:<8}"
        )

    print("\nConfusion matrix (rows = true class, cols = predicted class)")
    print("-" * 64)
    names = cm["label_names"]
    header = "  " + " " * 12 + "".join(f"{n:>12}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = cm["matrix"][i]
        print("  " + f"{name:<12}" + "".join(f"{v:>12}" for v in row))

    print(
        "\nNote: evaluation metrics only — not a clinical validation. "
        "This model is NOT certified as medically safe or suitable for "
        "clinical use."
    )


def save_metrics(metrics: dict, path: str | Path) -> Path:
    """Save metrics as JSON (machine-readable) and return the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # create artifacts/ if needed
    # indent=2 makes the file human-readable; JSON stays machine-readable too
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
