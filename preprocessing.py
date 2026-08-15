"""
preprocessing.py
----------------
All data-quality checks and cleaning steps for the Dry Bean pipeline live
here, so the preprocessing stage is a single, self-contained unit.

The stage has two parts:

  * `audit_dataset` - inspect the raw dataframe and report every quality
    signal (missing values, duplicates, non-numeric columns, constant
    columns, infinities, class balance, feature-scale spread). It only
    *reports*; it never changes the data.

  * `clean_dataset` - apply the cleaning actions the audit justifies:
    replace infinities, drop duplicate rows and drop constant (zero-variance)
    feature columns. Missing-value imputation and feature scaling are handled
    later by a scikit-learn Pipeline that is fit on the training split only
    (see `build_preprocessor`), so they cannot leak test information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def audit_dataset(frame: pd.DataFrame, target_column: str) -> dict:
    """Inspect the raw data and print a quality report. Returns the findings."""
    feats = [c for c in frame.columns if c != target_column]
    numeric = frame[feats].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in feats if c not in numeric]

    missing = int(frame.isna().sum().sum())
    duplicates = int(frame.duplicated().sum())
    constant_cols = [c for c in feats if frame[c].nunique(dropna=False) <= 1]
    infinities = (
        int(np.isinf(frame[numeric].to_numpy(dtype=float)).sum()) if numeric else 0
    )
    class_counts = frame[target_column].value_counts()
    imbalance_ratio = float(class_counts.max() / class_counts.min())

    print("\n" + "-" * 62)
    print("STAGE 1/2 - DATA-QUALITY AUDIT (no changes made yet)")
    print("-" * 62)
    print(f"  rows x cols              : {frame.shape[0]} x {frame.shape[1]}")
    print(f"  feature columns          : {len(feats)}")
    print(f"  missing values           : {missing}")
    print(f"  duplicate rows           : {duplicates}")
    print(f"  non-numeric features     : {non_numeric if non_numeric else 'none'}")
    print(f"  constant columns         : {constant_cols if constant_cols else 'none'}")
    print(f"  infinite values          : {infinities}")
    print(f"  classes                  : {len(class_counts)}")
    print(f"  class imbalance (max/min): {imbalance_ratio:.2f}")
    print("-" * 62)

    return {
        "rows": int(frame.shape[0]),
        "feature_columns": len(feats),
        "missing_values": missing,
        "duplicate_rows": duplicates,
        "non_numeric_features": non_numeric,
        "constant_columns": constant_cols,
        "infinite_values": infinities,
        "n_classes": int(len(class_counts)),
        "imbalance_ratio": round(imbalance_ratio, 3),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
    }


def clean_dataset(
    frame: pd.DataFrame,
    target_column: str,
    audit: dict,
    remove_duplicates: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply the cleaning steps justified by the audit. Returns (frame, actions)."""
    actions: list[str] = []
    frame = frame.copy()
    feats = [c for c in frame.columns if c != target_column]

    # 1) Replace any +/- infinity with NaN so the imputer can handle them later.
    if audit["infinite_values"] > 0:
        frame[feats] = frame[feats].replace([np.inf, -np.inf], np.nan)
        actions.append(f"replaced {audit['infinite_values']} infinite values with NaN")

    # 2) Drop exact duplicate rows.
    if remove_duplicates and audit["duplicate_rows"] > 0:
        before = len(frame)
        frame = frame.drop_duplicates().reset_index(drop=True)
        actions.append(f"dropped {before - len(frame)} duplicate rows")

    # 3) Drop constant (zero-variance) feature columns - they carry no signal.
    if audit["constant_columns"]:
        frame = frame.drop(columns=audit["constant_columns"])
        actions.append(f"dropped constant columns: {audit['constant_columns']}")

    print("STAGE 2/2 - CLEANING")
    if actions:
        for a in actions:
            print(f"  - {a}")
    else:
        print("  - no cleaning actions required")
    print(f"  result: {frame.shape[0]} rows x {frame.shape[1]} columns")
    print("  (missing-value imputation + scaling are bundled into each model's")
    print("   Pipeline and refit within every CV fold -> no leakage)")
    print("-" * 62)

    return frame, actions


def build_preprocessor() -> Pipeline:
    """Return the leakage-safe feature transformer: impute (median) -> scale.

    Fitting this on the training split only ensures the imputation statistics
    and scaling parameters never see the test data. The same fitted object is
    saved and reused by the Streamlit app on uploaded data.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
