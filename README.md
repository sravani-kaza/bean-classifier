# Dry Bean Variety Classifier

**Author:** KAZA SRAVANI &nbsp;|&nbsp; **Roll number:** 2025AC05278

An end-to-end machine-learning project that trains five classification models on the
**UCI Dry Bean** dataset and serves them through an interactive **Streamlit** web app.
The user uploads the held-out test CSV, picks a model, and inspects its Accuracy, AUC,
Precision, Recall, F1 and Matthews Correlation Coefficient together with a confusion
matrix and a full classification report.

---

## a. Problem statement

Given seven morphological (shape) measurements captured from images of dry beans, the
goal is to **automatically classify each bean into one of seven registered varieties**
(*Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira*). This is a supervised
**multi-class classification** problem. Reliable automatic grading of bean varieties
supports quality control and fair pricing in agricultural supply chains, where manual
sorting is slow and inconsistent.

## b. Dataset description

- **Source:** UCI Machine Learning Repository — *Dry Bean Dataset* (ID 602). Also
  mirrored on Kaggle.
- **Instances:** 13,611 raw rows (above the required minimum of 500). After removing 68
  exact duplicate rows found during the data-quality audit, **13,543 rows** are used.
- **Features:** 16 numeric attributes (above the required minimum of 12), all derived
  from bean images — `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`,
  `AspectRatio`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`,
  `Roundness`, `Compactness`, and four shape factors (`ShapeFactor1`–`ShapeFactor4`).
- **Target:** `Class` — 7 bean varieties (multi-class).
- **Class balance:** imbalanced, with a majority-to-minority ratio of about 6.8
  (Dermason ≈ 26% of rows, Bombay ≈ 3.8%).
- **Split & reproducibility:** a 75/25 stratified train/test split with a fixed random
  seed, so runs are reproducible.
- **Evaluation averaging:** because the target is multi-class, Precision, Recall and F1
  are reported with `weighted` averaging, and AUC is computed one-vs-rest.

### Data pre-processing & quality audit

All checks and cleaning live in a dedicated `preprocessing.py` module, so the
preprocessing stage is one self-contained unit with two parts:

- **Stage 1 — audit (`audit_dataset`):** inspects the raw data and reports missing
  values, non-numeric feature columns, constant columns, infinities, duplicate rows,
  class balance and feature-scale spread. It only reports; it changes nothing.
- **Stage 2 — cleaning (`clean_dataset`):** applies only the actions the audit
  justifies — here, dropping the 68 exact duplicate rows.

**Leakage-safe transform.** Median imputation and `StandardScaler` are bundled with each
classifier into a single scikit-learn `Pipeline` (`impute → scale → classifier`). It is
this whole pipeline that `GridSearchCV` tunes, so the imputer and scaler are refit inside
every cross-validation fold (on that fold's training data only) — no validation/test
information leaks into the transform. The preprocessing travels inside each saved model,
so the app passes raw uploaded features straight to the chosen model.

Audit findings on this dataset: **no missing values** (no imputation needed on the raw
data), **no non-numeric features** (no encoding needed beyond label-encoding the target),
**no constant or infinite values**, **68 duplicate rows removed**, an **imbalanced**
target (→ stratified split + weighted metrics), and widely differing **feature scales**
(→ standardization).

### Hyperparameter tuning

Each model is tuned with `GridSearchCV` using 3-fold stratified cross-validation on the
training split only (optimising weighted F1). The best configuration is refit and
evaluated once on the untouched test set. Grids searched:

| Model | Parameters searched |
|-------|---------------------|
| Logistic Regression | `C` ∈ {0.1, 1, 10} |
| Decision Tree | `max_depth` ∈ {8, 12, 20, None}, `min_samples_leaf` ∈ {1, 5, 10}, `criterion` ∈ {gini, entropy} |
| kNN | `n_neighbors` ∈ {5, 9, 15, 21}, `weights` ∈ {uniform, distance}, `p` ∈ {1, 2} |
| Naive Bayes | `var_smoothing` ∈ logspace(-11, -6) |
| Random Forest | `n_estimators` ∈ {200, 400}, `max_depth` ∈ {None, 20}, `min_samples_leaf` ∈ {1, 2} |

## c. GitHub Repository Link

**Repository:** https://github.com/sravani-kaza/bean-classifier

**Live Streamlit App:** https://bean-classifier-nxpxskwysecezk6b69s5ix.streamlit.app

## d. Models used

Five models are trained on the **same** dataset and split:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbours (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)

### Comparison table

**Test-set metrics** (held-out data):

| ML Model Name             | Accuracy |  AUC   | Precision | Recall |  F1    |  MCC   |
|---------------------------|:--------:|:------:|:---------:|:------:|:------:|:------:|
| Logistic Regression       | 0.9303   | 0.9938 | 0.9305    | 0.9303 | 0.9303 | 0.9157 |
| Decision Tree             | 0.9126   | 0.9798 | 0.9127    | 0.9126 | 0.9123 | 0.8943 |
| kNN                       | 0.9282   | 0.9912 | 0.9291    | 0.9282 | 0.9283 | 0.9132 |
| Naive Bayes               | 0.9022   | 0.9909 | 0.9038    | 0.9022 | 0.9025 | 0.8822 |
| Random Forest (Ensemble)  | 0.9318   | 0.9934 | 0.9321    | 0.9318 | 0.9317 | 0.9174 |

**Validation metrics** (mean over 3-fold cross-validation on the training data — the
scores used to select each model's hyperparameters):

| ML Model Name             | Accuracy |  AUC   | Precision | Recall |  F1    |  MCC   |
|---------------------------|:--------:|:------:|:---------:|:------:|:------:|:------:|
| Logistic Regression       | 0.9225   | 0.9933 | 0.9232    | 0.9225 | 0.9227 | 0.9063 |
| Decision Tree             | 0.9060   | 0.9767 | 0.9062    | 0.9060 | 0.9058 | 0.8863 |
| kNN                       | 0.9229   | 0.9910 | 0.9238    | 0.9229 | 0.9231 | 0.9068 |
| Naive Bayes               | 0.8935   | 0.9897 | 0.8948    | 0.8935 | 0.8936 | 0.8716 |
| Random Forest (Ensemble)  | 0.9208   | 0.9920 | 0.9210    | 0.9208 | 0.9208 | 0.9042 |

Validation and test scores track closely for every model, which indicates the models
generalise well and are not over-fitting.

### Observations

| ML Model Name            | Observation about model performance |
|--------------------------|-------------------------------------|
| Logistic Regression      | Very strong (test F1 0.930), essentially tied for the top. After standardisation the seven varieties are largely linearly separable, so a simple linear model competes with the ensemble. |
| Decision Tree            | Weakest of the tree-based models (F1 0.912). A single depth-limited tree keeps over-fitting in check, but on its own it captures less structure than the forest; close validation and test scores confirm it is not over-fitting. |
| kNN                      | Competitive with the leaders (F1 0.928) once features are scaled, showing that beans of the same variety form fairly tight clusters in the standardised feature space. |
| Naive Bayes              | Lowest overall (F1 0.903). Its feature-independence assumption is violated here — Area, Perimeter and the axis lengths are strongly correlated — which caps accuracy, although its AUC stays high (0.991). |
| Random Forest (Ensemble) | Best overall (F1 0.932, MCC 0.917). Averaging many decorrelated trees yields the most accurate and most robust predictions, and handles the class imbalance well. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — highest test F1 (0.9317) and MCC (0.9174). It edges out Logistic Regression (0.9303) and kNN (0.9283); the top three are effectively tied and the margin is within run-to-run variation, while Decision Tree and Naive Bayes are consistently behind. |

---

## Repository structure

```
project-folder/
├── app.py               # Streamlit front-end
├── train_models.py      # trains + evaluates the 5 models, saves artifacts
├── preprocessing.py     # data-quality audit + cleaning + transform pipeline
├── requirements.txt     # dependencies
├── README.md            # this file
├── test_data.csv        # exported held-out test set (generated by training)
└── model/               # saved artifacts (generated by training)
    ├── logistic_regression.pkl   # each .pkl is a full Pipeline:
    ├── decision_tree.pkl         #   impute -> scale -> classifier
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── label_encoder.pkl
    └── metrics.json
```

## How to run locally

```bash
# 1. Install dependencies (do this before training so versions match deployment)
pip install -r requirements.txt

# 2. Train the models — downloads the dataset once, runs the data-quality audit,
#    grid-search tuning for all 5 models, saves them + metrics.json, and exports
#    test_data.csv. The terminal shows live progress per model.
python train_models.py

# 3. Launch the web app
streamlit run app.py
```

The first training run downloads the dataset and caches it as `dry_bean.csv`. If the
machine has no internet access, download the **Dry Bean Dataset** from UCI/Kaggle, save
it as `dry_bean.csv` (16 feature columns + a `Class` column) in the project root, and
re-run step 2.

## Deploying on Streamlit Community Cloud

1. Push this repository (including the generated `model/` folder and `test_data.csv`) to GitHub.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New App**, select this repository, choose the `main` branch and `app.py`.
4. Under **Advanced settings**, select the same Python version you trained with.
5. Click **Deploy**. Once live, open the app and upload `test_data.csv` to see results.

## App features

- **CSV upload** — upload the held-out `test_data.csv`.
- **Model dropdown** — switch between all five trained models.
- **Live metrics** — Accuracy, AUC, Precision, Recall, F1, MCC on the uploaded data.
- **Confusion matrix** and **classification report** for the selected model.
- **Comparison tab** — test and validation metrics side by side for all five models, a
  validation-vs-test F1 chart, and the best hyperparameters chosen for each model.
- Download predictions as a CSV.
