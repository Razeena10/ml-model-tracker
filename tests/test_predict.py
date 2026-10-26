"""Tests for predict.py — run with: pytest -v

Models are fitted inside the tests themselves (on the real dataset), so
these tests are self-contained: they pass with or without artifacts/model.joblib.
"""

import joblib
import numpy as np
import pytest

from predict import load_trained_model, predict_rows
from src.data_loader import load_dataset
from train import build_pipeline, split_data


@pytest.fixture(scope="module")
def fitted():
    """Fit one pipeline per test session (cheap, real dataset)."""
    X, y = load_dataset()
    X_train, X_test, y_train, _ = split_data(X, y)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline, X_test


def test_load_saved_model_roundtrip(tmp_path, fitted):
    """joblib save -> load must return a usable pipeline with .predict."""
    pipeline, X_test = fitted
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    loaded = load_trained_model(path)
    preds = predict_rows(loaded, X_test.head(5))
    assert len(preds) == 5
    assert set(np.unique(preds)).issubset({0, 1})


def test_saved_model_matches_in_memory(fitted):
    """Requirement 5: saved-model predictions == in-memory predictions."""
    pipeline, X_test = fitted
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "model.joblib"
        joblib.dump(pipeline, path)
        loaded = load_trained_model(path)
        in_memory = pipeline.predict(X_test)
        from_disk = predict_rows(loaded, X_test)
        assert (in_memory == from_disk).all()


def test_single_row_1d_input_accepted(fitted):
    """A single flat row of 30 features must be reshaped to (1, 30)."""
    pipeline, X_test = fitted
    one_row = X_test.iloc[0].to_numpy()
    assert one_row.shape == (30,)
    pred = predict_rows(pipeline, one_row)
    assert pred.shape == (1,)


def test_wrong_feature_count_raises_clear_error(fitted):
    """Edge case: 29 features must raise ValueError with a clear message."""
    pipeline, _ = fitted
    bad_row = np.zeros(29)
    with pytest.raises(ValueError, match="Expected 30 features"):
        predict_rows(pipeline, bad_row)


def test_empty_input_raises(fitted):
    """Edge case: empty input must raise, not silently return nothing."""
    pipeline, _ = fitted
    with pytest.raises(ValueError, match="empty"):
        predict_rows(pipeline, np.empty((0, 30)))


def test_missing_model_file_raises(tmp_path):
    """Loading a nonexistent file gives a helpful FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Run 'python train.py'"):
        load_trained_model(tmp_path / "nope.joblib")
