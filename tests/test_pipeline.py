"""Tests for train.build_pipeline — the scikit-learn Pipeline used for training.

These verify the pipeline's structure (so the model stays the intended
StandardScaler -> LogisticRegression baseline) and that fitting it produces a
usable, correctly-shaped classifier. They also tie the pipeline to the
evaluator to confirm both class-specific metric groups are produced from
real predictions.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataset
from src.evaluate import compute_metrics
from train import MAX_ITER, build_pipeline, split_data


@pytest.fixture(scope="module")
def split():
    """One stratified split reused across this module (fast, real dataset)."""
    X, y = load_dataset()
    return split_data(X, y)  # X_train, X_test, y_train, y_test


def test_build_pipeline_returns_pipeline():
    """build_pipeline must return a scikit-learn Pipeline."""
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)


def test_pipeline_has_scaler_and_classifier_steps():
    """The pipeline must be exactly StandardScaler -> LogisticRegression."""
    pipe = build_pipeline()
    assert list(pipe.named_steps) == ["scaler", "classifier"]
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    assert isinstance(pipe.named_steps["classifier"], LogisticRegression)


def test_classifier_uses_documented_max_iter():
    """max_iter must match the documented 1000 (avoids convergence warnings)."""
    pipe = build_pipeline()
    assert pipe.named_steps["classifier"].max_iter == MAX_ITER == 1000


def test_scaler_is_unfitted_until_pipeline_fit(split):
    """A fresh pipeline must not have a fitted scaler (no leakage before fit)."""
    X_train, X_test, y_train, _ = split
    pipe = build_pipeline()
    scaler = pipe.named_steps["scaler"]
    # An unfitted StandardScaler has no learned statistics yet.
    assert not hasattr(scaler, "mean_") and not hasattr(scaler, "var_")


def test_fit_then_predict_produces_valid_labels(split):
    """After fit, predict must return a label array of the right shape/contents."""
    X_train, X_test, y_train, y_test = split
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    assert preds.shape[0] == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})


def test_scaler_is_fitted_after_pipeline_fit(split):
    """Fitting the pipeline must fit the scaler on TRAINING data only."""
    X_train, X_test, y_train, _ = split
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    scaler = pipe.named_steps["scaler"]
    # A fitted StandardScaler learns per-feature mean/variance.
    assert hasattr(scaler, "mean_")
    assert hasattr(scaler, "var_")
    assert scaler.mean_.shape[0] == X_train.shape[1]


def test_pipeline_predictions_yield_both_metric_groups(split):
    """End-to-end: predictions -> compute_metrics -> both groups present and sane.

    This checks that the evaluator's two positive-class groups are produced
    from real pipeline output and lie in [0, 1], without hardcoding values.
    """
    X_train, X_test, y_train, y_test = split
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    metrics = compute_metrics(y_test, pipe.predict(X_test))

    groups = metrics["metrics"]
    assert set(groups) == {"benign_positive", "malignant_positive"}
    for name in ("precision", "recall", "f1"):
        for grp in groups.values():
            assert 0.0 <= grp[name] <= 1.0
    # accuracy is a single global metric shared by both groups
    assert 0.0 <= metrics["accuracy"] <= 1.0
