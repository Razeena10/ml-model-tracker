"""Shared pytest fixtures for the ml-model-tracker test suite.

The ``isolated_mlflow`` fixture redirects MLflow to a temporary SQLite
database and artifact directory under pytest's ``tmp_path`` (the system temp
dir, outside the project and OneDrive).  This avoids:

  * polluting the project's ``mlflow.db`` or ``mlartifacts/``,
  * lock contention from any running ``mlflow ui`` server processes,
  * OneDrive file-sync interference with ``log_model``.

Any test that needs MLflow should use this fixture.  Tests that only need
sklearn/pandas (the majority) import nothing from here.
"""

from __future__ import annotations

import mlflow
import pytest
from mlflow.tracking import MlflowClient

import src.tracking as tracking_module


@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    """Redirect MLflow to an isolated temp SQLite DB and artifact directory.

    Yields (uri, experiment_name) so callers can assert on the URI.

    Teardown resets global MLflow state so nothing leaks between tests.
    """
    # End any active run left from a previous (possibly failed) test.
    mlflow.end_run()

    db = tmp_path / "test_mlflow.db"
    art = (tmp_path / "mlartifacts").as_posix()
    uri = f"sqlite:///{db.as_posix()}"
    exp_name = "isolated-test-experiment"

    # Patch the module-level constants so setup_experiment() and any code
    # that reads TRACKING_URI/ARTIFACT_LOCATION/EXPERIMENT_NAME directly
    # all see the isolated values.
    monkeypatch.setattr(tracking_module, "TRACKING_URI", uri)
    monkeypatch.setattr(tracking_module, "ARTIFACT_LOCATION", art)
    monkeypatch.setattr(tracking_module, "EXPERIMENT_NAME", exp_name)

    # Point the global MLflow client at the temp store and pre-create the
    # experiment with the explicit artifact location so model artifacts
    # never fall back to the default relative mlruns/ path.
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    try:
        client.create_experiment(exp_name, artifact_location=art)
    except Exception:
        pass  # already exists on a reused tmp_path (shouldn't happen, but safe)

    yield uri, exp_name

    # Teardown: end any active run and reset global MLflow state.
    mlflow.end_run()
    mlflow.set_tracking_uri("")
