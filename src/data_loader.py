"""Data loading and inspection for the Breast Cancer Wisconsin dataset.

This module is the single source of truth for loading data in this project.
Later steps (train.py, evaluation, tests) should import ``load_dataset``
instead of calling ``sklearn.datasets.load_breast_cancer`` themselves.

Target labels (the "truth" we try to predict):
    0 = malignant  (cancerous)
    1 = benign     (not cancerous)

Which class is "positive" for evaluation?
    By scikit-learn's convention for this dataset, class 1 (benign) is the
    positive class. That means precision/recall/F1 later in the project will
    describe how well we detect BENIGN cases, unless we set pos_label=0.
    In a real clinical setting you would usually flip this, because catching
    malignancy (class 0) is the medically important error to avoid. We keep
    scikit-learn's default here for reproducibility and will state the
    pos_label explicitly during the evaluation step.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer

# Human-readable class names, keyed by the integer labels used in the data.
TARGET_NAMES: dict[int, str] = {0: "malignant", 1: "benign"}

# The class treated as "positive" by sklearn's convention for this dataset.
POSITIVE_CLASS: int = 1  # 1 = benign


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load features and target as a (DataFrame, Series) pair.

    Returns
    -------
    X : pandas.DataFrame   shape (569, 30) — one column per measurement
    y : pandas.Series       shape (569,)    — name "target", values 0 or 1
    """
    # fetch the built-in dataset (no downloads, ships with scikit-learn)
    bunch = load_breast_cancer()

    # Features -> DataFrame so we get labeled columns (e.g. "mean radius")
    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)

    # Target -> Series with a clear name, so merged tables stay readable
    y = pd.Series(bunch.target, name="target")

    # Fail fast with a clear message if features and labels ever disagree
    validate_rows(X, y)
    return X, y


def validate_rows(X: pd.DataFrame, y: pd.Series) -> None:
    """Raise ValueError if X and y don't describe the same number of rows."""
    if len(X) != len(y):
        raise ValueError(
            f"Feature/label mismatch: X has {len(X)} rows but y has {len(y)}."
        )
    if len(X) == 0:
        raise ValueError("Dataset is empty: refusing to continue with 0 rows.")


def inspect_dataset(X: pd.DataFrame, y: pd.Series) -> None:
    """Print a beginner-friendly summary of the dataset."""
    print("=" * 60)
    print("BREAST CANCER WISCONSIN DATASET — INSPECTION")
    print("=" * 60)

    # 1. Overall shape: (rows, columns)
    print(f"\nShape: {X.shape[0]} samples x {X.shape[1]} features")

    # 2. First five rows, transposed so all 30 features are readable
    print("\nFirst 5 rows (transposed):")
    print(X.head().T)

    # 3. Class balance: how many malignant (0) vs benign (1)
    print("\nTarget class counts:")
    for label, count in y.value_counts().sort_index().items():
        print(f"  {label} ({TARGET_NAMES[label]}): {count}")

    # 4. The 30 feature column names
    print(f"\nFeature names ({len(X.columns)}):")
    print(", ".join(X.columns))

    # 5. Missing values per column — this dataset should have none
    total_missing = int(X.isna().sum().sum())
    print(f"\nTotal missing values: {total_missing}")


if __name__ == "__main__":
    # Allows: python -m src.data_loader   (handy for a quick manual look)
    features, target = load_dataset()
    inspect_dataset(features, target)
