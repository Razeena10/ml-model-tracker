"""Metric calculation for the classification pipeline (Step 4).

Positive class — and why it matters
-----------------------------------
``POSITIVE_CLASS = 1`` (benign) is sklearn's convention for this dataset,
imported here from ``src.data_loader`` so there is ONE source of truth.

Precision and recall are defined *with respect to a chosen positive class*:

* **Precision** = "of the cases we CALLED positive, how many really were?"
* **Recall**    = "of the cases that REALLY WERE positive, how many did we catch?"

Flip the positive class and the two numbers swap meaning: predicting
"malignant" for everything gives 100% recall for malignancy but terrible
precision. That's why the label must be stated explicitly, never assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_loader import POSITIVE_CLASS


def compute_metrics(y_true, y_pred, pos_label: int = POSITIVE_CLASS) -> dict:
    """Compute evaluation metrics for a binary classifier.

    ``zero_division=0`` is passed to precision/recall/F1: if a degenerate
    model predicts no positive cases at all, the precision denominator would
    otherwise be zero and sklearn would warn/return NaN. Returning 0 keeps
    results JSON-serializable and honest ("we never predicted positive").
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        # rows = true labels, cols = predicted labels, order [0, 1]:
        # [[TN, FP], [FN, TP]] when pos_label=1
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "positive_class": int(pos_label),
        "n_test_samples": int(len(y_true)),
    }


def print_metrics(metrics: dict) -> None:
    """Print the metric dict in a human-readable format."""
    print("\nTEST SET EVALUATION")
    print("-" * 40)
    print(f"Positive class : {metrics['positive_class']}")
    print(f"Test samples   : {metrics['n_test_samples']}")
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"Precision      : {metrics['precision']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"F1-score       : {metrics['f1']:.4f}")
    print("Confusion matrix (rows=true, cols=predicted, order [0, 1]):")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")


def save_metrics(metrics: dict, path: str | Path) -> Path:
    """Save metrics as JSON (machine-readable) and return the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # create artifacts/ if needed
    # indent=2 makes the file human-readable; JSON stays machine-readable too
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
