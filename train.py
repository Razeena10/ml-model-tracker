"""Baseline training + evaluation with MLflow tracking (Steps 3–5).

What is a "baseline"?
    The simplest reasonable model, trained first, so every later, fancier
    model has a number to beat.

This script keeps ONE training/evaluation path; MLflow tracking wraps it
via the helpers in src/tracking.py (no duplicated logic).

Run with:  python train.py
UI with:   mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from pathlib import Path

import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import POSITIVE_CLASS, load_dataset
from src.evaluate import compute_metrics, print_metrics, save_metrics
from src.tracking import (
    TRACKING_URI,
    log_metrics_dict,
    log_parameters,
    log_run_metadata,
    log_trained_model,
    setup_experiment,
)

# Fixed seed => every run produces the exact same split => reproducible results
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20% of the data is held out for testing, 80% for training

# max_iter=1000 gives the solver plenty of iterations to converge
# (the default of 100 can trigger convergence warnings on some datasets)
MAX_ITER = 1000


def split_data(X, y):
    """Split into train/test sets: stratified, seeded, test data untouched."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # keep the malignant/benign ratio identical in both sets
    )


def build_pipeline() -> Pipeline:
    """Build the model pipeline: scale features, then classify.

    Why a Pipeline instead of scaling separately?
      * ``StandardScaler.fit_transform`` is only ever called on TRAINING
        data as the pipeline flows through — the test set is only ever
        transformed, never fitted on. That prevents data leakage.
      * One object to save/predict with later, instead of two loose pieces.
    """
    return Pipeline(
        steps=[
            # scale features to mean=0, std=1 (LogisticRegression likes this)
            ("scaler", StandardScaler()),
            # the classic, fast, interpretable linear classifier
            ("classifier", LogisticRegression(max_iter=MAX_ITER)),
        ]
    )


def main() -> None:
    # 1. Load data (reuses the validated loader from Step 2)
    X, y = load_dataset()

    # 2. Train/test split — stratified and seeded
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    try:
        # Everything inside this `with` block is ONE MLflow run.
        # On any exception, MLflow marks the run FAILED automatically
        # and the except-block below reports it clearly (requirement 7).
        setup_experiment()
        with mlflow.start_run(run_name="logistic-regression-baseline") as active_run:
            # 3. Build and fit the pipeline on TRAINING data only.
            #    .fit() is the only place learning happens; X_test/y_test are
            #    deliberately NOT passed in, so the model never sees test labels.
            pipeline = build_pipeline()
            pipeline.fit(X_train, y_train)

            # 4. Confirm completion (the trained pipeline is ready for evaluation)
            print(f"Training completed. Model type: {type(pipeline.named_steps['classifier']).__name__}")

            # 5. Evaluate on the held-out test set.
            #    .predict() only reads from the model; the test set is used solely
            #    for scoring — never for fitting or tuning.
            y_pred = pipeline.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            print_metrics(metrics)

            # 6. Save machine-readable results under artifacts/ (git-ignored)
            out_path = save_metrics(metrics, Path("artifacts") / "metrics.json")
            print(f"\nMetrics saved to: {out_path}")

            # 6b. Serialize the FITTED pipeline with joblib so predict.py
            #     can reuse it without retraining. joblib (not pickle) is
            #     sklearn's recommended format for models with numpy arrays.
            model_path = Path("artifacts") / "model.joblib"
            joblib.dump(pipeline, model_path)
            print(f"Model saved to: {model_path}")

            # 6c. Reproducibility check: the reloaded model must produce
            #     IDENTICAL predictions to the in-memory model on the test set.
            #     If this ever prints False, serialization is broken.
            from predict import load_trained_model, predict_rows  # local import

            reloaded = load_trained_model(model_path)
            disk_preds = predict_rows(reloaded, X_test)
            match = bool((disk_preds == y_pred).all())
            print(f"Saved-model predictions match in-memory model: {match}")
            if not match:
                raise RuntimeError(
                    "Saved model predictions differ from in-memory predictions."
                )

            # 7. Track with MLflow: inputs (params), outputs (metrics),
            #    the trained model (artifact), and context (tags).
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
                dataset_name="sklearn.datasets.load_breast_cancer (Breast Cancer Wisconsin)",
                positive_class=POSITIVE_CLASS,
                n_train_samples=len(X_train),
                n_test_samples=len(X_test),
            )

            run_id = active_run.info.run_id
            print(f"\nMLflow run recorded: {run_id}")
            print(f"Tracking store: {TRACKING_URI}")
            print("View it with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    except Exception as exc:
        # Report the failure clearly, then re-raise so the process
        # still exits non-zero (CI and shells can detect it).
        print(f"\nERROR: MLflow run failed — {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
