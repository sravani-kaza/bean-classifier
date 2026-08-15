"""
app.py
------
Interactive Streamlit front-end for the Dry Bean classifier.

The user uploads the held-out test CSV (produced by `train_models.py`),
picks one of the five trained models from a dropdown, and the app reports
Accuracy / AUC / Precision / Recall / F1 / MCC together with a confusion
matrix and a full classification report.

Run locally with:   streamlit run app.py

Author: KAZA SRAVANI (2025AC05278)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = Path(__file__).resolve().parent / "model"

st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide",
)

# A little custom styling so the app does not look like a bare template.
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem;}
        div[data-testid="stMetricValue"] {font-size: 1.6rem;}
        .app-header {
            border-left: 6px solid #7a4b2b; padding-left: 14px; margin-bottom: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Loading cached artifacts
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load every trained model pipeline plus the label encoder and metrics.

    Each model .pkl is a full scikit-learn Pipeline (impute -> scale ->
    classifier), so no separate preprocessor needs to be loaded.
    """
    with open(MODEL_DIR / "metrics.json") as fh:
        meta = json.load(fh)

    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

    models = {}
    for slug, info in meta["models"].items():
        models[info["display_name"]] = joblib.load(MODEL_DIR / f"{slug}.pkl")

    return meta, encoder, models


def score_predictions(y_true, y_pred, y_proba, class_labels, averaging):
    """Return the six evaluation metrics as a dict (binary + multi-class safe)."""
    if len(class_labels) == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average=averaging, labels=class_labels
        )
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=averaging, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=averaging, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=averaging, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Page body
# --------------------------------------------------------------------------- #

st.markdown('<h1 class="app-header">🫘 Dry Bean Variety Classifier</h1>', unsafe_allow_html=True)
st.caption(
    "Upload the test CSV, choose a model, and inspect its performance. "
    "Five models were trained on the UCI Dry Bean dataset (16 shape features, 7 varieties)."
)

try:
    meta, encoder, models = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Please run `python train_models.py` first so "
        "that the `model/` directory contains the trained `.pkl` files and `metrics.json`."
    )
    st.stop()

feature_names = meta["feature_names"]
target_column = meta["target_column"]
averaging = meta["averaging"]
class_labels = np.arange(len(encoder.classes_))

# ---- Sidebar controls ---------------------------------------------------- #
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
    chosen_model_name = st.selectbox("Choose a model", list(models.keys()))
    st.divider()
    st.markdown(
        f"**Target column:** `{target_column}`  \n"
        f"**Classes:** {len(encoder.classes_)}  \n"
        f"**Averaging:** `{averaging}`"
    )

tab_predict, tab_compare = st.tabs(["🔎 Evaluate a model", "📊 Model comparison"])

# =========================================================================== #
# Tab 1 - live evaluation on the uploaded CSV
# =========================================================================== #
with tab_predict:
    if uploaded is None:
        st.info("️Upload the `test_data.csv` file from the sidebar to begin.")
    else:
        data = pd.read_csv(uploaded)
        st.subheader("Preview of uploaded data")
        st.dataframe(data.head(10), width='stretch')

        missing = [c for c in feature_names if c not in data.columns]
        if missing:
            st.error(f"The uploaded file is missing expected feature columns: {missing}")
            st.stop()

        # Each model is a full pipeline (impute -> scale -> classifier),
        # so raw feature columns can be passed straight in.
        X = data[feature_names].to_numpy(dtype=float)

        model = models[chosen_model_name]
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        pred_labels = encoder.inverse_transform(y_pred)

        has_truth = target_column in data.columns
        if has_truth:
            y_true = encoder.transform(data[target_column].astype(str))
            scores = score_predictions(y_true, y_pred, y_proba, class_labels, averaging)

            st.subheader(f"Evaluation metrics — {chosen_model_name}")
            cols = st.columns(6)
            for col, (name, value) in zip(cols, scores.items()):
                col.metric(name, f"{value:.4f}")

            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Confusion matrix**")
                cm = confusion_matrix(y_true, y_pred, labels=class_labels)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="YlOrBr",
                    xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax,
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)

            with right:
                st.markdown("**Classification report**")
                report = classification_report(
                    y_true, y_pred, target_names=encoder.classes_,
                    zero_division=0, output_dict=True,
                )
                st.dataframe(pd.DataFrame(report).transpose().round(3),
                             width='stretch')
        else:
            st.warning(
                f"No `{target_column}` column found, so metrics cannot be computed. "
                "Showing predictions only."
            )

        st.subheader("Predictions")
        result = data.copy()
        result["Predicted"] = pred_labels
        st.dataframe(result.head(50), width='stretch')
        st.download_button(
            "Download predictions as CSV",
            result.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

# =========================================================================== #
# Tab 2 - static comparison table at training time
# =========================================================================== #
with tab_compare:
    metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

    st.subheader("Test-set metrics for all models")
    test_rows = []
    for info in meta["models"].values():
        test_rows.append({"ML Model": info["display_name"],
                          **{m: info[m] for m in metric_cols}})
    test_table = pd.DataFrame(test_rows)
    st.dataframe(test_table, width='stretch', hide_index=True)

    st.subheader("Validation metrics (cross-validation on training data)")
    st.caption(
        f"Mean over {meta.get('cv_folds', '?')} CV folds — the scores used to "
        "select each model's hyperparameters before touching the test set."
    )
    val_rows = []
    for info in meta["models"].values():
        val = info.get("validation", {})
        val_rows.append({"ML Model": info["display_name"],
                         **{m: val.get(m) for m in metric_cols}})
    val_table = pd.DataFrame(val_rows)
    st.dataframe(val_table, width='stretch', hide_index=True)

    best = test_table.loc[test_table["F1"].idxmax(), "ML Model"]
    st.success(f"Overall winner — highest test F1: **{best}**")

    st.markdown("**Validation vs test F1 by model**")
    f1_compare = pd.DataFrame({
        "ML Model": test_table["ML Model"],
        "Validation F1": val_table["F1"],
        "Test F1": test_table["F1"],
    }).set_index("ML Model")
    st.bar_chart(f1_compare)

    with st.expander("Best hyperparameters chosen by cross-validated grid search"):
        st.caption(
            f"Each model was tuned with {meta.get('cv_folds', '?')}-fold "
            "cross-validation, optimising weighted F1."
        )
        for info in meta["models"].values():
            params = info.get("best_params", {})
            st.markdown(f"**{info['display_name']}** — `{params}`")
