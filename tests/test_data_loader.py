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

# ---------------------------------------------------------------------------
# Additional loading & feature/label consistency checks
# ---------------------------------------------------------------------------


def test_target_values_are_only_binary():
    """Labels must be exactly the two documented integer classes (0 and 1)."""
    _, y = load_dataset()
    assert set(y.unique()) == {0, 1}


def test_target_series_has_expected_name():
    """The target Series keeps a readable name so joined tables stay clear."""
    _, y = load_dataset()
    assert y.name == "target"


def test_features_are_all_numeric():
    """All 30 features must be numeric (the pipeline scales them)."""
    X, _ = load_dataset()
    assert len(X.columns) == 30
    # Numeric dtype for every column (int/float/bool all count as numeric)
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)


def test_no_duplicate_feature_columns():
    """Feature names must be unique so column lookups are unambiguous."""
    X, _ = load_dataset()
    assert X.columns.is_unique


def test_features_and_labels_are_index_aligned():
    """X and y must share the same index (row i of X matches row i of y)."""
    X, y = load_dataset()
    assert X.index.equals(y.index)
    assert len(X) == len(y) == 569


def test_validate_rows_rejects_empty_dataset():
    """validate_rows must refuse an empty feature/label pair."""
    empty_X = pd.DataFrame()
    empty_y = pd.Series([], dtype=int)
    with pytest.raises(ValueError, match="empty"):
        validate_rows(empty_X, empty_y)
