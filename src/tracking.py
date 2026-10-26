"""MLflow tracking helpers (Step 5).

MLflow vocabulary, in plain terms:
- **Experiment**: a named collection of runs — like a project folder.
  Here: one experiment per project, "breast-cancer-classification".
- **Run**: one execution of the training pipeline, with a unique run ID.
- **Parameter**: an INPUT we chose (algorithm, seed, test size) — "what we tried".
- **Metric**: an OUTPUT we measured (accuracy, F1) — "how it did".
- **Artifact**: a FILE the run produced (the saved model) — "what we kept".

This module owns every MLflow call so ``train.py`` stays about *training*
and the ML code stays testable without a tracking server.

Privacy note: this project tracks a public, built-in dataset only.
Never log patient identifiers, credentials, or anything sensitive.
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import sklearn

# Local tracking store: a SQLite database file inside the project (git-ignored).
# MLflow 3.x deprecated the plain "file:./mlruns" folder backend (it is in
# maintenance mode and raises unless MLFLOW_ALLOW_FILE_STORE=true), so we use
# the recommended local database backend: still just a file, no server needed.
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "breast-cancer-classification"
MODEL_ARTIFACT_NAME = "model"

# Only these scalar metrics get logged to MLflow (confusion_matrix is a
# list, not a scalar, and MLflow metrics must be numbers).
METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


def setup_experiment() -> None:
    """Point MLflow at the local store; create the experiment or reuse it."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_parameters(params: dict) -> None:
    """Record INPUT settings. MLflow params are stored as strings."""
    mlflow.log_params({key: str(value) for key, value in params.items()})


def log_metrics_dict(metrics: dict) -> None:
    """Record the four scalar evaluation metrics."""
    mlflow.log_metrics({name: metrics[name] for name in METRIC_NAMES})


def log_trained_model(pipeline) -> None:
    """Log the fitted sklearn pipeline with the sklearn-model logging API.

    In MLflow 3.x the artifact is addressed with ``name=`` (the old
    ``artifact_path=`` argument is deprecated).
    """
    mlflow.sklearn.log_model(sk_model=pipeline, name=MODEL_ARTIFACT_NAME)


def log_run_metadata(dataset_name: str, **extra_tags) -> None:
    """Record non-sensitive context: dataset name and library versions."""
    mlflow.set_tag("dataset", dataset_name)
    mlflow.set_tag("sklearn_version", sklearn.__version__)
    for key, value in extra_tags.items():
        mlflow.set_tag(key, str(value))
