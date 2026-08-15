"""
train_models.py
---------------
Trains and evaluates five classification models on the UCI "Dry Bean"
dataset (16 numeric shape features, 7 bean varieties, 13,611 samples), with
cross-validated hyperparameter tuning (GridSearchCV) for every model.

  1. Load the dataset (fetched once from the UCI ML Repository and cached
     locally as `dry_bean.csv`).
  2. Split into train / test partitions (stratified).
  3. Fit a StandardScaler and a LabelEncoder on the training partition.
  4. For each model, run a GridSearchCV over a small parameter grid using
     k-fold cross-validation on the TRAINING data only, and keep the best
     estimator (selected by weighted F1).
  5. Score every tuned model on the held-out TEST set with Accuracy,
     ROC-AUC (one-vs-rest), Precision, Recall, F1 and Matthews Correlation
     Coefficient.
  6. Persist the fitted best models, the scaler, the label encoder and a
     `metrics.json` summary (including the chosen hyperparameters) into the
     `model/` directory.
  7. Export the held-out test partition to `test_data.csv`.

Author: KAZA SRAVANI (2025AC05278)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from preprocessing import audit_dataset, build_preprocessor, clean_dataset

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RANDOM_SEED = 1            # change this to get a slightly different split
TEST_FRACTION = 0.25
REMOVE_DUPLICATES = True    # drop exact duplicate rows found during the audit
AVERAGING = "weighted"     # how per-class Precision/Recall/F1 are combined
CV_FOLDS = 3               # folds used inside GridSearchCV (raise to 5 if you like)
TUNING_SCORE = "f1_weighted"   # metric the grid search optimises
SEARCH_VERBOSE = 2         # GridSearchCV verbosity: 0 silent, 1 light, 2 per-fit

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
DATA_CACHE = PROJECT_ROOT / "dry_bean.csv"
TEST_EXPORT = PROJECT_ROOT / "test_data.csv"
TARGET_COLUMN = "Class"

MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_dataset() -> pd.DataFrame:
    """Return the Dry Bean dataset as a single dataframe (features + target).

    On the first run the data is pulled from the UCI repository and written
    to `dry_bean.csv`. Every later run reads the cached copy, so the script
    also works on machines without internet access once the cache exists.
    """
    if DATA_CACHE.exists():
        print(f"Reading cached dataset from {DATA_CACHE.name}")
        return pd.read_csv(DATA_CACHE)

    print("Cache not found -> downloading Dry Bean dataset from UCI ...")
    from ucimlrepo import fetch_ucirepo

    bundle = fetch_ucirepo(id=602)          # 602 == Dry Bean Dataset
    features = bundle.data.features
    target = bundle.data.targets

    frame = pd.concat([features, target], axis=1)
    frame = frame.rename(columns={frame.columns[-1]: TARGET_COLUMN})
    frame.to_csv(DATA_CACHE, index=False)
    print(f"Saved a local cache -> {DATA_CACHE.name} ({len(frame)} rows)")
    return frame


# --------------------------------------------------------------------------- #
# Metric container
# --------------------------------------------------------------------------- #

@dataclass
class ScoreCard:
    """Holds the six required metrics for one trained model."""

    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    mcc: float

    def as_dict(self) -> dict:
        return {
            "Accuracy": round(self.accuracy, 4),
            "AUC": round(self.auc, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1": round(self.f1, 4),
            "MCC": round(self.mcc, 4),
        }


def evaluate(y_true, y_pred, y_proba, class_labels) -> ScoreCard:
    """Compute all six metrics, transparently handling the multi-class case."""
    if len(class_labels) == 2:
        # binary: AUC uses the probability of the positive class
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        # multi-class: one-vs-rest AUC averaged across classes
        auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average=AVERAGING,
            labels=class_labels,
        )

    return ScoreCard(
        accuracy=accuracy_score(y_true, y_pred),
        auc=auc,
        precision=precision_score(y_true, y_pred, average=AVERAGING, zero_division=0),
        recall=recall_score(y_true, y_pred, average=AVERAGING, zero_division=0),
        f1=f1_score(y_true, y_pred, average=AVERAGING, zero_division=0),
        mcc=matthews_corrcoef(y_true, y_pred),
    )


# --------------------------------------------------------------------------- #
# Search spaces: (base estimator, parameter grid) per model
# --------------------------------------------------------------------------- #

def build_search_spaces() -> dict:
    """Return each estimator paired with the grid GridSearchCV will explore."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_SEED),
            {
                "C": [0.1, 1.0, 10.0],
                "solver": ["lbfgs"],
            },
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=RANDOM_SEED),
            {
                "max_depth": [8, 12, 20, None],
                "min_samples_leaf": [1, 5, 10],
                "criterion": ["gini", "entropy"],
            },
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [5, 9, 15, 21],
                "weights": ["uniform", "distance"],
                "p": [1, 2],           # 1 = Manhattan, 2 = Euclidean
            },
        ),
        "naive_bayes": (
            GaussianNB(),
            {
                "var_smoothing": np.logspace(-11, -6, 6),
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 20],
                "min_samples_leaf": [1, 2],
            },
        ),
    }


PRETTY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "kNN",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest (Ensemble)",
}


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #

def main() -> None:
    frame = load_dataset()
    print(f"Dataset shape: {frame.shape[0]} rows x {frame.shape[1]} columns")

    # ===================== PREPROCESSING STAGE ============================= #
    # Stage 1: audit the raw data (checks only).
    audit = audit_dataset(frame, TARGET_COLUMN)
    # Stage 2: apply the cleaning actions the audit justifies.
    frame, clean_actions = clean_dataset(frame, TARGET_COLUMN, audit, REMOVE_DUPLICATES)
    # (Imputation + scaling happen just below, fit on the training split only.)
    # ====================================================================== #

    feature_names = [c for c in frame.columns if c != TARGET_COLUMN]
    X = frame[feature_names].to_numpy(dtype=float)
    y_raw = frame[TARGET_COLUMN].astype(str).to_numpy()

    # Encode string class labels -> integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_labels = np.arange(len(encoder.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=RANDOM_SEED, stratify=y
    )

    # NOTE: features are NOT scaled here. Instead, the impute->scale steps are
    # bundled with each classifier into one Pipeline (built below), so the
    # transform is refit inside every cross-validation fold -> no leakage.

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    metrics_summary: dict = {}

    # Multiple scorers so the same cross-validation that tunes each model also
    # yields all six VALIDATION metrics for free (read from cv_results_).
    validation_scorers = {
        "Accuracy": "accuracy",
        "AUC": "roc_auc_ovr_weighted",
        "Precision": "precision_weighted",
        "Recall": "recall_weighted",
        "F1": "f1_weighted",
        "MCC": make_scorer(matthews_corrcoef),
    }

    spaces = build_search_spaces()
    total_models = len(spaces)
    run_start = time.time()

    for idx, (slug, (estimator, grid)) in enumerate(spaces.items(), start=1):
        n_combos = int(np.prod([len(v) for v in grid.values()]))
        print(
            f"\n[{idx}/{total_models}] Tuning {PRETTY_NAMES[slug]} "
            f"-> {n_combos} param combos x {CV_FOLDS} folds "
            f"= {n_combos * CV_FOLDS} fits",
            flush=True,
        )
        t0 = time.time()

        # One Pipeline per model: impute -> scale -> classifier. GridSearchCV
        # refits the whole pipeline on each fold, so scaling/imputation never
        # see the validation fold. Grid keys are prefixed to target the "clf" step.
        pipe = Pipeline(steps=[("prep", build_preprocessor()), ("clf", estimator)])
        pipe_grid = {f"clf__{k}": v for k, v in grid.items()}

        search = GridSearchCV(
            pipe,
            param_grid=pipe_grid,
            scoring=validation_scorers,
            refit="F1",      # pick the best config by validation weighted-F1
            cv=cv,
            n_jobs=-1,
            verbose=SEARCH_VERBOSE,   # live per-fit progress in the terminal
        )
        search.fit(X_train, y_train)          # raw (unscaled) features
        best_model = search.best_estimator_   # full pipeline (prep + clf)

        # VALIDATION metrics = mean CV score of the winning config (no extra cost)
        bi = search.best_index_
        validation = {
            name: round(float(search.cv_results_[f"mean_test_{name}"][bi]), 4)
            for name in validation_scorers
        }

        # TEST metrics = tuned pipeline scored once on the untouched test set
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)
        card = evaluate(y_test, y_pred, y_proba, class_labels)

        # Strip the "clf__" prefix so the saved hyperparameters read cleanly
        clean_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

        joblib.dump(best_model, MODEL_DIR / f"{slug}.pkl")
        metrics_summary[slug] = {
            "display_name": PRETTY_NAMES[slug],
            "best_params": clean_params,
            "validation": validation,   # cross-validated metrics on training data
            **card.as_dict(),           # test-set metrics (top level, back-compatible)
        }

        elapsed = time.time() - t0
        print(f"   [{idx}/{total_models} done] best params: {clean_params}",
              flush=True)
        print(f"   validation: " + "  ".join(f"{k}={v}" for k, v in validation.items()),
              flush=True)
        print("   test:       " + "  ".join(f"{k}={v}" for k, v in card.as_dict().items()),
              flush=True)
        print(f"   (model took {elapsed:.1f}s, total {time.time() - run_start:.1f}s)",
              flush=True)

    # Persist the label encoder + metadata (each model .pkl is a full pipeline
    # that already contains its own fitted impute->scale preprocessing).
    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")

    def _clean(obj):
        """Make numpy scalars JSON-serialisable."""
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    with open(MODEL_DIR / "metrics.json", "w") as fh:
        json.dump(
            _clean(
                {
                    "target_column": TARGET_COLUMN,
                    "feature_names": feature_names,
                    "class_names": list(encoder.classes_),
                    "averaging": AVERAGING,
                    "random_seed": RANDOM_SEED,
                    "cv_folds": CV_FOLDS,
                    "tuning_score": TUNING_SCORE,
                    "data_audit": audit,
                    "cleaning_actions": clean_actions,
                    "duplicates_removed": REMOVE_DUPLICATES,
                    "models": metrics_summary,
                }
            ),
            fh,
            indent=2,
        )

    # Export the untouched (unscaled) test partition for the Streamlit app
    test_frame = pd.DataFrame(X_test, columns=feature_names)
    test_frame[TARGET_COLUMN] = encoder.inverse_transform(y_test)
    test_frame.to_csv(TEST_EXPORT, index=False)

    # ----------------------------------------------------------------------- #
    # Consolidated summary report (validation vs test) for all five models
    # ----------------------------------------------------------------------- #
    metric_order = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

    val_rows, test_rows, combined_rows = [], [], []
    for info in metrics_summary.values():
        name = info["display_name"]
        val_rows.append({"ML Model": name, **info["validation"]})
        test_rows.append({"ML Model": name, **{m: info[m] for m in metric_order}})
        combined_rows.append(
            {"ML Model": name, "Split": "validation (CV)", **info["validation"]}
        )
        combined_rows.append(
            {"ML Model": name, "Split": "test", **{m: info[m] for m in metric_order}}
        )

    val_table = pd.DataFrame(val_rows)[["ML Model"] + metric_order]
    test_table = pd.DataFrame(test_rows)[["ML Model"] + metric_order]
    combined = pd.DataFrame(combined_rows)[["ML Model", "Split"] + metric_order]

    combined.to_csv(MODEL_DIR / "summary_report.csv", index=False)

    def df_to_md(df: pd.DataFrame) -> str:
        """Render a dataframe as a GitHub-flavoured markdown table (no deps)."""
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join("---" for _ in df.columns) + " |"
        body = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)
        ]
        return "\n".join([header, sep, *body])

    winner = test_table.loc[test_table["F1"].idxmax(), "ML Model"]
    md_lines = [
        "# Model Summary Report",
        "",
        f"Dataset target: `{TARGET_COLUMN}` | classes: {len(encoder.classes_)} | "
        f"averaging: `{AVERAGING}` | CV folds: {CV_FOLDS} | seed: {RANDOM_SEED}",
        "",
        "## Validation metrics (mean over cross-validation folds, training data)",
        "",
        df_to_md(val_table),
        "",
        "## Test metrics (held-out test set)",
        "",
        df_to_md(test_table),
        "",
        f"**Overall winner (highest test F1): {winner}**",
        "",
    ]
    (MODEL_DIR / "summary_report.md").write_text("\n".join(md_lines))

    print("\n" + "=" * 70)
    print("VALIDATION metrics (cross-validation on training data)")
    print("=" * 70)
    print(val_table.to_string(index=False))
    print("\n" + "=" * 70)
    print("TEST metrics (held-out test set)")
    print("=" * 70)
    print(test_table.to_string(index=False))
    print(f"\nOverall winner (highest test F1): {winner}")

    print(f"\nSaved 5 tuned model pipelines (impute->scale->clf) + encoder "
          f"to '{MODEL_DIR.name}/'")
    print(f"Wrote summary report -> '{MODEL_DIR.name}/summary_report.csv' and "
          f"'{MODEL_DIR.name}/summary_report.md'")
    print(f"Exported test partition ({len(test_frame)} rows) to '{TEST_EXPORT.name}'")
    print("Done.")


if __name__ == "__main__":
    main()
