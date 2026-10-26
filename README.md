# ml-model-tracker

A reproducible binary-classification pipeline built with **scikit-learn** and
tracked with **MLflow**, complete with automated tests. Built as a Week 1
portfolio project for ML Engineer / MLOps roles.

## Objectives

- Train a baseline classifier end-to-end on a trusted, built-in dataset.
- Guarantee **reproducibility**: fixed seeds, stratified splits, pinned
  dependency versions, and automated tests that enforce both.
- Track parameters, metrics, and the trained model with **MLflow**, so
  experiments are comparable and auditable instead of lost in terminal scrollback.
- Practice engineering hygiene: modular code, input validation, a clean
  `.gitignore`, and honest documentation (every number below comes from a
  real, reproducible run).

## Architecture and workflow

```
             ┌──────────────────────────────────────────────────────┐
  data ───▶  │ src/data_loader.py  load + validate (569×30, 0 NaNs) │
             └──────────────┬───────────────────────────────────────┘
                            ▼
             ┌──────────────────────────────────────────────────────┐
             │ train.py  stratified split (80/20, seed=42)          │
             │   Pipeline: StandardScaler → LogisticRegression      │
             │   .fit() on TRAINING data only                       │
             └──────────────┬───────────────────────────────────────┘
                            ▼
             ┌──────────────────────────────────────────────────────┐
             │ src/evaluate.py  accuracy / precision / recall / F1  │
             │   + confusion matrix; saved to artifacts/metrics.json│
             └──────────────┬───────────────────────────────────────┘
                            ▼
        ┌───────────────────┴────────────────────┐
        ▼                                        ▼
┌─────────────────────────┐        ┌──────────────────────────────┐
│ src/tracking.py (MLflow)│        │ artifacts/model.joblib       │
│  params, metrics, model │        │   consumed by predict.py     │
│  → sqlite:///mlflow.db  │        │   (loads & predicts, no fit) │
└─────────────────────────┘        └──────────────────────────────┘
```

The test set is held out from every fitting step; preprocessing lives inside
the sklearn `Pipeline` so the scaler is only ever fitted on training data.

## Dataset

- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset, loaded from
  `sklearn.datasets.load_breast_cancer` — **built into scikit-learn**, so no
  external download and no third-party data host. It originates from the
  UCI Machine Learning Repository and was donated by W. Wolberg.
- **Size:** 569 samples × 30 numeric features (mean, error, and "worst"
  measurements of cell-nuclei characteristics from fine-needle aspirates).
- **Target labels:**
  - `0` = **malignant** (cancerous)
  - `1` = **benign** (not cancerous)
- **Positive class for evaluation: `1` (benign)** — this is scikit-learn's
  convention for this dataset, and the code states it explicitly rather than
  relying on defaults. In a clinical setting you would usually flip this to
  `pos_label=0`, because missing a malignancy is the dangerous error.

## Prerequisites

- Python 3.14 (developed and tested on 3.14.3)
- pip 25+
- Git (optional, for cloning)

## Installation

```bash
# 1. Clone the repository (replace with your URL after pushing)
git clone <your-repo-url>
cd ml-model-tracker

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS / Linux

# 4. Install pinned dependencies
pip install -r requirements.txt
```

## Usage — exact commands

```bash
# Train + evaluate + track with MLflow + save artifacts/
python train.py

# Load artifacts/model.joblib and predict on sample rows (never retrains)
python predict.py

# Run the automated test suite (19 tests)
pytest -v

# Open the MLflow UI to browse experiments and runs
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → open http://127.0.0.1:5000
```

> If `mlflow` is not found after activating the venv, call it by path:
> `.venv/Scripts/mlflow ui --backend-store-uri sqlite:///mlflow.db` (Windows)
> or `.venv/bin/mlflow ui ...` (macOS / Linux).

## Evaluation results

From a verified run of `python train.py` (held-out test set, **114 samples**,
positive class = `1` benign):

| Metric    | Value  | Interpretation |
|-----------|--------|----------------|
| Accuracy  | 0.9825 | 98.25% of all test predictions were correct (112/114) |
| Precision | 0.9861 | Of cases predicted *benign*, 98.61% truly were |
| Recall    | 0.9861 | Of truly *benign* cases, 98.61% were correctly identified |
| F1-score  | 0.9861 | Harmonic mean of precision and recall |

**Confusion matrix** (rows = true class, columns = predicted, order `[0, 1]`):

```
[[41,  1],   ← 41 malignant correct,  1 misclassified as benign
 [ 1, 71]]   ←  1 benign misclassified as malignant, 71 correct
```

**Why not accuracy alone?** The dataset is 62.7% benign, so a lazy
"always predict benign" model scores 0.627 accuracy while catching zero
malignancies. Precision, recall, F1, and the confusion matrix expose what
accuracy hides: *which* mistakes a model makes. Here, the model made exactly
2 errors out of 114 — one in each direction.

## MLflow: experiments, runs, and artifacts

This project uses a **local SQLite tracking store** (`mlflow.db`, git-ignored).
MLflow 3.x deprecated the plain `mlruns/` folder backend, so a database
backend is used instead — still just a file, no server required.

- **Experiment** — a named collection of runs. This project uses
  `breast-cancer-classification`.
- **Run** — one execution of the training pipeline, with a unique run ID
  recorded for auditability.
- **Parameter** — an *input* you chose ("what we tried"): algorithm, scaler,
  `random_state`, `test_size`, `max_iter`.
- **Metric** — an *output* you measured ("how it did"): accuracy, precision,
  recall, F1.
- **Artifact** — a *file the run produced*: the trained sklearn `Pipeline`
  logged with `mlflow.sklearn.log_model`, plus tags recording the dataset
  name and sklearn version.

Every run prints its run ID; browse them at http://127.0.0.1:5000 after
starting the UI command above.

## Model serialization

The fitted pipeline is saved to `artifacts/model.joblib` with **joblib**.
`predict.py` loads it and predicts — it never retrains.

- **Trust warning:** joblib files use pickle internally. Loading a `.joblib`
  file can execute arbitrary code. Only load model files you trained
  yourself or from sources you fully trust.
- **Version sensitivity:** pickled models record the library versions that
  created them. Loading with a different (especially older) scikit-learn may
  fail or behave unexpectedly. This project pins versions in
  `requirements.txt` (built with scikit-learn 1.9.1).

## Tests

19 tests enforce the project's guarantees (all passing, warnings treated as
errors):

```bash
pytest -v
```

- `tests/test_data_loader.py` — dataset shape, class balance, no missing
  values, row-count validation (including a failing-input case).
- `tests/test_evaluate.py` — metrics verified against hand-calculated
  values; the `zero_division=0` edge case; JSON round-trip.
- `tests/test_predict.py` — joblib save/load round-trip; saved-model
  predictions exactly match the in-memory model; invalid-input errors.
- `tests/test_reproducibility.py` — same seed ⇒ identical splits and
  identical metrics across repeated full training runs.

## Project structure

```
ml-model-tracker/
├── src/
│   ├── data_loader.py    # Load + validate the dataset (single source of truth)
│   ├── evaluate.py       # Metric calculation, printing, JSON export
│   └── tracking.py       # All MLflow calls (experiment/run/params/metrics/model)
├── tests/                # 19 pytest tests (loader, evaluation, prediction, reproducibility)
├── artifacts/            # Generated: metrics.json, model.joblib (git-ignored)
├── train.py              # Entry point: split → fit → evaluate → track → save
├── predict.py            # Load artifacts/model.joblib and predict (no training)
├── requirements.txt      # Pinned dependency versions
├── mlflow.db             # Local MLflow store (git-ignored, regenerated)
└── README.md
```

## Limitations, reproducibility, and responsible use

**Limitations**
- A single baseline model (LogisticRegression) with default hyperparameters;
  no cross-validation or hyperparameter search.
- Evaluation on a single 20% hold-out split; the test set has 114 samples, so
  metrics carry noticeable statistical uncertainty.
- The positive class is `benign` (sklearn convention), which is arguably the
  backwards choice for a cancer-screening task.

**Reproducibility**
- Fixed `random_state=42` and stratified splitting; LogisticRegression on
  this data is deterministic, so repeated runs give byte-identical metrics
  (enforced by `tests/test_reproducibility.py`).
- Dependency versions are pinned in `requirements.txt`; MLflow tags each run
  with the sklearn version.

**Data-leakage precautions**
- Preprocessing (`StandardScaler`) lives inside the sklearn `Pipeline`, so it
  is fitted on training data only.
- The test set is used exclusively for final scoring — never for fitting or
  tuning — and the saved model is verified to match the in-memory model on
  those held-out rows.

**Responsible use**
- This model is a **teaching artifact, not a medical device**. It must never
  be used to inform real clinical decisions.
- The dataset contains no patient identifiers, and nothing sensitive is
  logged to MLflow — keep it that way in any extension of this project.

## Future improvements

- Add cross-validation (e.g., `cross_val_score`) for more robust estimates.
- Try alternative models (Random Forest, Gradient Boosting) and compare them
  in MLflow against this baseline.
- Log an evaluation curve or feature-importance artifact.
- Register the best model in the MLflow Model Registry.
- Add a CI workflow (GitHub Actions) that runs `pytest` on every push.
- Offer a `pos_label=0` variant so recall reflects malignancy detection.

## License & attribution

Dataset: Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository,
via `sklearn.datasets`. Add a repository license of your choice before
publishing.
