"""Tests for src/tracking.py — MLflow logging against an ISOLATED temp store.

The ``isolated_mlflow`` fixture (defined in conftest.py) redirects MLflow to
a temporary SQLite database and artifact directory under pytest's ``tmp_path``
(system temp dir, outside the project and OneDrive).

All tests here call the project's own tracking helpers with NO mocking of the
logging behaviour — they verify what was actually persisted by reading it back
through the MLflow API.
"""

from __future__ import annotations

import os
import pathlib

import mlflow
import pytest
import sklearn
from mlflow.tracking import MlflowClient

import src.tracking as tracking_module
from src.data_loader import POSITIVE_CLASS, load_dataset
from src.evaluate import compute_metrics
from src.tracking import (
    METRIC_NAMES,
    log_metrics_dict,
    log_parameters,
    log_run_metadata,
    log_trained_model,
    setup_experiment,
)
from train import MAX_ITER, RANDOM_STATE, TEST_SIZE, build_pipeline, split_data


# ---------------------------------------------------------------------------
# Configuration and setup_experiment
# ---------------------------------------------------------------------------


def test_setup_experiment_targets_isolated_store(isolated_mlflow):
    """setup_experiment must point MLflow at the configured store/experiment."""
    uri, exp_name = isolated_mlflow
    setup_experiment()

    assert mlflow.get_tracking_uri() == uri
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    assert exp is not None, "experiment was not created/reused by setup_experiment"


def test_setup_experiment_creates_with_explicit_artifact_location(isolated_mlflow, tmp_path):
    """setup_experiment must create the experiment with an explicit artifact location.

    This ensures model artifacts land in the configured ARTIFACT_LOCATION
    (the absolute mlartifacts/ path) rather than the relative mlruns/ default
    which caused the OneDrive-sync hang.
    """
    uri, exp_name = isolated_mlflow
    setup_experiment()

    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    assert exp is not None
    # The artifact location must be non-empty and NOT the relative mlruns/ default.
    art_loc = exp.artifact_location
    assert art_loc, "experiment has no artifact_location"
    # Must not use the bare relative 'mlruns' path that triggers the OneDrive hang.
    assert "mlruns" not in art_loc.lower() or os.path.isabs(art_loc.replace("file:///", "")), (
        f"artifact_location {art_loc!r} points into relative mlruns/ tree"
    )


def test_tracking_uri_is_absolute_by_default():
    """The default TRACKING_URI derived from the project root must be absolute.

    A relative URI like sqlite:///mlflow.db resolves differently depending on
    CWD and can map into an OneDrive-synced path, causing log_model hangs.
    """
    uri = tracking_module.TRACKING_URI
    # Strip the sqlite:/// scheme prefix to get the bare path
    if uri.startswith("sqlite:///"):
        bare = uri[len("sqlite:///"):]
        assert os.path.isabs(bare), (
            f"TRACKING_URI bare path {bare!r} is relative — "
            "it must be absolute to avoid CWD-dependent behaviour"
        )


def test_artifact_location_is_absolute_by_default():
    """ARTIFACT_LOCATION must be an absolute path by default."""
    loc = tracking_module.ARTIFACT_LOCATION
    # Strip file:// prefix if present
    bare = loc.replace("file:///", "").replace("file://", "")
    assert os.path.isabs(bare) or os.path.isabs(loc), (
        f"ARTIFACT_LOCATION {loc!r} is relative"
    )


def test_tracking_uri_env_var_override(monkeypatch, tmp_path):
    """MLFLOW_TRACKING_URI env-var must override the default TRACKING_URI.

    This is the recommended way to redirect logging in tests or CI without
    modifying source.
    """
    override = f"sqlite:///{(tmp_path / 'override.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", override)

    # Re-import the module constant after patching the env-var
    import importlib
    import src.tracking as t
    importlib.reload(t)

    assert t.TRACKING_URI == override

    # Restore (monkeypatch cleans up env-var; reload the original module state)
    importlib.reload(t)


# ---------------------------------------------------------------------------
# End-to-end logging: params, metrics, model, tags
# ---------------------------------------------------------------------------


def test_mlflow_logging_records_params_metrics_model_and_tags(isolated_mlflow, tmp_path):
    """One real run must log params, the four scalar metrics, the model, and tags.

    Reads the run back through the MLflow API to verify what was persisted.
    """
    _uri, _exp_name = isolated_mlflow

    # Real data + real training so logged values are genuine.
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    setup_experiment()
    with mlflow.start_run(run_name="test-tracking-run") as run:
        run_id = run.info.run_id
        log_parameters(
            {
                "algorithm": "LogisticRegression",
                "scaler": "StandardScaler",
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "max_iter": MAX_ITER,
            }
        )
        log_metrics_dict(metrics)
        log_trained_model(pipeline)
        log_run_metadata(
            dataset_name="test-dataset",
            positive_class=POSITIVE_CLASS,
            n_train_samples=len(X_train),
            n_test_samples=len(X_test),
        )

    client = MlflowClient()
    finished = client.get_run(run_id)
    data = finished.data

    # Run completed normally.
    assert finished.info.status == "FINISHED"

    # --- Parameters ---
    assert data.params["algorithm"] == "LogisticRegression"
    assert data.params["scaler"] == "StandardScaler"
    assert data.params["random_state"] == str(RANDOM_STATE)
    assert data.params["test_size"] == str(TEST_SIZE)
    assert data.params["max_iter"] == str(MAX_ITER)

    # --- Metrics: four scalars; confusion_matrix must NOT appear ---
    for name in METRIC_NAMES:
        assert name in data.metrics, f"metric {name!r} was not logged"
        assert isinstance(data.metrics[name], float)
    assert data.metrics["accuracy"] == pytest.approx(metrics["accuracy"])
    assert data.metrics["f1"] == pytest.approx(metrics["f1"])
    assert "confusion_matrix" not in data.metrics

    # --- Tags ---
    assert data.tags["dataset"] == "test-dataset"
    assert data.tags["sklearn_version"] == sklearn.__version__
    assert data.tags["positive_class"] == str(POSITIVE_CLASS)
    assert data.tags["n_train_samples"] == str(len(X_train))
    assert data.tags["n_test_samples"] == str(len(X_test))

    # --- Model artifact: MLmodel file must exist in the isolated store ---
    art_root = pathlib.Path(tmp_path / "mlartifacts")
    mlmodel_files = list(art_root.rglob("MLmodel"))
    assert mlmodel_files, (
        "log_trained_model did not persist a model artifact under the "
        "isolated artifact store"
    )


def test_failed_run_is_marked_failed_not_finished(isolated_mlflow):
    """A run that raises inside the context must be FAILED, not FINISHED.

    This ensures errors are not silently presented as successful tracking.
    """
    setup_experiment()
    run_id = None
    try:
        with mlflow.start_run(run_name="deliberately-failing-run") as run:
            run_id = run.info.run_id
            log_parameters({"algorithm": "LogisticRegression"})
            raise RuntimeError("Simulated training failure")
    except RuntimeError:
        pass  # expected

    assert run_id is not None
    client = MlflowClient()
    finished = client.get_run(run_id)
    assert finished.info.status == "FAILED", (
        f"Expected FAILED status but got {finished.info.status!r} — "
        "errors must not be silently presented as successful tracking"
    )


def test_log_metrics_dict_only_logs_scalar_metrics(isolated_mlflow):
    """log_metrics_dict must log only the four METRIC_NAMES scalars."""
    setup_experiment()
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    metrics = compute_metrics(
        y_test, build_pipeline().fit(X_train, y_train).predict(X_test)
    )
    with mlflow.start_run() as run:
        log_metrics_dict(metrics)

    data = MlflowClient().get_run(run.info.run_id).data
    assert set(data.metrics.keys()) == set(METRIC_NAMES)
    # The complex sub-keys must not appear as flattened scalars
    for forbidden in ("confusion_matrix", "n_test_samples", "positive_class"):
        assert forbidden not in data.metrics
