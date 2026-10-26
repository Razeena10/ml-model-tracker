"""Tests for src/data_loader.py — run with: pytest -v"""

import pandas as pd
import pytest

from src.data_loader import POSITIVE_CLASS, TARGET_NAMES, load_dataset, validate_rows


def test_load_dataset_returns_dataframe_and_series():
    """Loader must return (features DataFrame, target Series)."""
    X, y = load_dataset()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_dataset_shape_is_569_by_30():
    """The built-in dataset is fixed: 569 samples, 30 features."""
    X, y = load_dataset()
    assert X.shape == (569, 30)
    assert len(y) == 569


def test_features_and_labels_have_matching_rows():
    """Requirement 5: row counts must agree."""
    X, y = load_dataset()
    assert len(X) == len(y)


def test_target_classes_and_counts():
    """Known class balance: 212 malignant (0), 357 benign (1)."""
    _, y = load_dataset()
    counts = y.value_counts().to_dict()
    assert counts == {0: 212, 1: 357}
    # sanity check on our documented label names / positive class
    assert TARGET_NAMES == {0: "malignant", 1: "benign"}
    assert POSITIVE_CLASS == 1


def test_no_missing_values_and_named_columns():
    """Dataset ships clean, and columns keep sklearn's meaningful names."""
    X, _ = load_dataset()
    assert X.isna().sum().sum() == 0
    assert len(X.columns) == 30
    assert "mean radius" in X.columns


def test_validate_rows_raises_on_mismatch():
    """validate_rows must fail loudly if X and y lengths differ."""
    X, y = load_dataset()
    with pytest.raises(ValueError, match="mismatch"):
        validate_rows(X, y.iloc[:-1])  # drop one label on purpose
