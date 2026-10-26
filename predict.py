"""Load the saved model and predict on new feature rows (Step 6).

Ordinary prediction NEVER retrains: this module only calls ``joblib.load``
and ``pipeline.predict``. Fitting happens exclusively in ``train.py``.

SECURITY — joblib files are pickle files
----------------------------------------
``joblib.load`` uses Python's pickle machinery underneath, and unpickling
a malicious file can execute arbitrary code on your machine. Therefore:

  * Only load ``.joblib`` files you trained yourself, or from a source
    you trust completely.
  * Never download model files from strangers on the internet.

COMPATIBILITY — dependency versions matter
------------------------------------------
A pickled scikit-learn model records the library versions that created it.
Loading it with a different (especially older) scikit-learn version can
fail or silently change behavior. The saved model expects the versions
pinned in requirements.txt (scikit-learn 1.9.1).

Run directly for a demo:  python predict.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_dataset

# Where train.py saves the fitted pipeline
DEFAULT_MODEL_PATH = Path("artifacts") / "model.joblib"

# The Breast Cancer dataset always has 30 features; predictions must match
EXPECTED_N_FEATURES = 30


def load_trained_model(path: str | Path = DEFAULT_MODEL_PATH):
    """Load a saved sklearn pipeline from disk. No training happens here."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run 'python train.py' first."
        )
    return joblib.load(path)


def predict_rows(model, rows) -> np.ndarray:
    """Predict labels for new feature rows after validating their shape.

    Parameters
    ----------
    model : a fitted sklearn pipeline (from ``load_trained_model``)
    rows  : array-like of shape (n_samples, 30) — same feature order
            and units as the training data.

    Raises
    ------
    ValueError if the feature count is wrong or the input is empty.
    """
    # If the caller passed a pandas DataFrame, keep it as-is: the pipeline
    # was fitted with named columns, and sklearn warns when feature names
    # disappear. Raw arrays/lists take the numpy path instead.
    if hasattr(rows, "columns"):
        data = rows
        n_cols = data.shape[1]
    else:
        # Convert to a 2D numpy array; a single row [f1..f30] becomes (1, 30)
        arr = np.asarray(rows, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n_cols = arr.shape[1] if arr.ndim == 2 else 0

        # Validate BEFORE wrapping in a DataFrame, so a wrong-width input
        # gets OUR clear message instead of a confusing pandas error.
        if n_cols == 0:
            raise ValueError("Input is empty: provide at least one row of features.")
        if n_cols != EXPECTED_N_FEATURES:
            raise ValueError(
                f"Expected {EXPECTED_N_FEATURES} features per row, got {n_cols}. "
                "Feature order must match the training data "
                "(see src/data_loader.py for the 30 column names)."
            )

        # Wrap raw arrays in a DataFrame with the model's own feature names
        # (fitted estimators remember them via ``feature_names_in_``) so
        # sklearn stays happy and column order is validated implicitly.
        names = getattr(model, "feature_names_in_", None)
        data = pd.DataFrame(arr, columns=names) if names is not None else arr

    if len(data) == 0:
        raise ValueError("Input is empty: provide at least one row of features.")

    # Same validation for DataFrame inputs (arrays were checked above)
    if hasattr(rows, "columns") and data.shape[1] != EXPECTED_N_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_N_FEATURES} features per row, got {data.shape[1]}. "
            "Feature order must match the training data "
            "(see src/data_loader.py for the 30 column names)."
        )

    # .predict() only reads the fitted parameters — no retraining (req. 7)
    return model.predict(data)


if __name__ == "__main__":
    # Demo: load the saved model, predict on 3 rows it has never scored
    # from a fresh call, and show the predicted class names.
    from src.data_loader import TARGET_NAMES

    model = load_trained_model()
    X, _ = load_dataset()
    sample = X.head(3)
    preds = predict_rows(model, sample)
    print(f"Model loaded: {type(model).__name__}")
    for i, (idx, p) in enumerate(zip(sample.index, preds)):
        print(f"  row {idx}: predicted class {p} ({TARGET_NAMES[int(p)]})")
