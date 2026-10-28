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

Paths and the OneDrive / CWD problem
-------------------------------------
MLflow's default relative URI ``sqlite:///mlflow.db`` is resolved at runtime
relative to the *current working directory*.  When the project lives under
OneDrive and ``mlflow ui`` servers are running, the relative ``mlruns/``
artifact directory falls inside the OneDrive tree, causing ``log_model`` to
hang indefinitely because OneDrive locks files being synced.

To fix this:
  * ``TRACKING_URI`` is derived from the absolute path of THIS file so the
    SQLite DB is always resolved to the project root, not the caller's CWD.
  * ``ARTIFACT_LOCATION`` points to ``mlartifacts/`` beside the DB (also
    absolute), so model artifacts never land in the relative ``mlruns/`` tree.
  * Both can be overridden at runtime via the ``MLFLOW_TRACKING_URI`` /
    ``MLFLOW_ARTIFACT_LOCATION`` environment variables — this is the
    recommended way for tests and CI to use an isolated temporary store.

Override example (isolated test run, outside OneDrive):
    MLFLOW_TRACKING_URI=sqlite:////tmp/test.db python train.py

Privacy note: this project tracks a public, built-in dataset only.
Never log patient identifiers, credentials, or anything sensitive.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import sklearn

# ---------------------------------------------------------------------------
# Paths — resolved to absolute so behaviour is CWD-independent
# ---------------------------------------------------------------------------

# The project root is the directory that contains THIS source file's parent
# (src/ -> project root).  Resolving here means the DB is always at
# <project_root>/mlflow.db, no matter where the caller runs from.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Allow full URI override via env-var — useful for tests and CI that need an
# isolated, temporary store outside the project / OneDrive directory.
# Default: absolute path to the project-local SQLite file.
TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + (_PROJECT_ROOT / "mlflow.db").as_posix(),
)

# Artifact storage location — also absolute and overridable.
# Placing it beside mlflow.db (not under the relative mlruns/ default) avoids
# OneDrive locking when log_model writes model files.
ARTIFACT_LOCATION: str = os.environ.get(
    "MLFLOW_ARTIFACT_LOCATION",
    (_PROJECT_ROOT / "mlartifacts").as_posix(),
)

EXPERIMENT_NAME: str = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "breast-cancer-classification",
)

MODEL_ARTIFACT_NAME = "model"

# Only these scalar metrics get logged to MLflow (confusion_matrix is a
# list, not a scalar, and MLflow metrics must be numbers).
METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


def setup_experiment() -> None:
    """Point MLflow at the configured store; create the experiment or reuse it.

    The experiment is created with an explicit ``artifact_location`` so that
    model artifacts are written to the absolute ``ARTIFACT_LOCATION`` path
    (outside the relative ``mlruns/`` tree that previously triggered
    OneDrive locking).  If the experiment already exists its location is
    kept unchanged.
    """
    mlflow.set_tracking_uri(TRACKING_URI)

    # Create the experiment with an explicit artifact location the first time;
    # get_experiment_by_name returns None if it does not exist yet.
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)

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
