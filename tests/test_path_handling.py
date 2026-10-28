"""Tests for path handling and output-directory reliability.

Verifies that:
  * train.py resolves artifact paths absolutely (CWD-independent).
  * The artifacts/ directory is created when missing.
  * predict.py's DEFAULT_MODEL_PATH is absolute.
  * load_trained_model works when given an absolute path.
  * predict_rows produces valid output from a model loaded by absolute path.
  * Useful errors are raised for missing/wrong-path model files.
"""

from __future__ import annotations

import os
import pathlib

import joblib
import numpy as np
import pytest

import predict as predict_module
from predict import load_trained_model, predict_rows
from src.data_loader import load_dataset
from train import ARTIFACTS_DIR, build_pipeline, split_data


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_artifacts_dir_is_absolute():
    """train.ARTIFACTS_DIR must be an absolute path (CWD-independent)."""
    assert ARTIFACTS_DIR.is_absolute(), (
        f"ARTIFACTS_DIR {ARTIFACTS_DIR!r} is relative; "
        "train.py must resolve it from the project root so it works from any CWD"
    )


def test_artifacts_dir_points_inside_project():
    """ARTIFACTS_DIR must sit inside the project root (not a temp dir or system path)."""
    project_root = pathlib.Path(__file__).resolve().parent.parent
    assert ARTIFACTS_DIR.is_relative_to(project_root), (
        f"ARTIFACTS_DIR {ARTIFACTS_DIR} is not under the project root {project_root}"
    )


def test_default_model_path_is_absolute():
    """predict.DEFAULT_MODEL_PATH must be absolute (CWD-independent)."""
    assert predict_module.DEFAULT_MODEL_PATH.is_absolute(), (
        f"DEFAULT_MODEL_PATH {predict_module.DEFAULT_MODEL_PATH!r} is relative; "
        "predict.py must resolve it from the project root"
    )


def test_default_model_path_is_inside_project_artifacts():
    """DEFAULT_MODEL_PATH must live inside artifacts/ under the project root."""
    project_root = pathlib.Path(__file__).resolve().parent.parent
    assert predict_module.DEFAULT_MODEL_PATH.is_relative_to(project_root / "artifacts"), (
        f"DEFAULT_MODEL_PATH {predict_module.DEFAULT_MODEL_PATH} is not under artifacts/"
    )


# ---------------------------------------------------------------------------
# Output directory creation
# ---------------------------------------------------------------------------


def test_save_metrics_creates_artifacts_dir_when_missing(tmp_path):
    """save_metrics must create an arbitrary nested output directory."""
    from src.evaluate import compute_metrics, save_metrics

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    metrics = compute_metrics(y_test, build_pipeline().fit(X_train, y_train).predict(X_test))
    nested = tmp_path / "new" / "nested" / "dir" / "metrics.json"
    assert not nested.parent.exists()
    out = save_metrics(metrics, nested)
    assert out.exists()
    assert out == nested


def test_train_artifacts_dir_can_be_created(tmp_path, monkeypatch):
    """ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True) must not raise
    even when the directory does not exist yet."""
    fake_artifacts = tmp_path / "brand_new_artifacts"
    assert not fake_artifacts.exists()
    monkeypatch.setattr("train.ARTIFACTS_DIR", fake_artifacts)
    # This is the exact call train.main() makes before saving outputs.
    fake_artifacts.mkdir(parents=True, exist_ok=True)
    assert fake_artifacts.exists()


# ---------------------------------------------------------------------------
# Model loading by absolute path
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    """Fit a pipeline and joblib-dump it to a tmp_path absolute path."""
    tmp = tmp_path_factory.mktemp("model")
    X, y = load_dataset()
    X_train, _, y_train, _ = split_data(X, y)
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    path = tmp / "model.joblib"
    joblib.dump(pipe, path)
    return pipe, path


def test_load_trained_model_accepts_absolute_path(saved_model):
    """load_trained_model must succeed with an absolute pathlib.Path."""
    _pipe, path = saved_model
    assert path.is_absolute()
    loaded = load_trained_model(path)
    assert loaded is not None
    assert hasattr(loaded, "predict")


def test_load_trained_model_accepts_absolute_string_path(saved_model):
    """load_trained_model must succeed with an absolute str path."""
    _pipe, path = saved_model
    loaded = load_trained_model(str(path))
    assert hasattr(loaded, "predict")


def test_load_trained_model_missing_file_gives_clear_error(tmp_path):
    """Loading a nonexistent absolute path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Run 'python train.py'"):
        load_trained_model(tmp_path / "does_not_exist.joblib")


def test_predict_rows_from_loaded_model_produces_valid_output(saved_model):
    """predict_rows on a model loaded by absolute path must return {0,1} labels."""
    pipe, path = saved_model
    loaded = load_trained_model(path)
    X, _ = load_dataset()
    preds = predict_rows(loaded, X.head(10))
    assert preds.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_predictions_from_absolute_path_match_in_memory(saved_model):
    """Predictions from a model loaded by absolute path must match in-memory."""
    pipe, path = saved_model
    loaded = load_trained_model(path)
    X, _ = load_dataset()
    sample = X.head(20)
    assert (pipe.predict(sample) == predict_rows(loaded, sample)).all()
