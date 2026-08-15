# Model Summary Report

Dataset target: `Class` | classes: 7 | averaging: `weighted` | CV folds: 3 | seed: 1

## Validation metrics (mean over cross-validation folds, training data)

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9225 | 0.9933 | 0.9232 | 0.9225 | 0.9227 | 0.9063 |
| Decision Tree | 0.906 | 0.9767 | 0.9062 | 0.906 | 0.9058 | 0.8863 |
| kNN | 0.9229 | 0.991 | 0.9238 | 0.9229 | 0.9231 | 0.9068 |
| Naive Bayes | 0.8935 | 0.9897 | 0.8948 | 0.8935 | 0.8936 | 0.8716 |
| Random Forest (Ensemble) | 0.9208 | 0.992 | 0.921 | 0.9208 | 0.9208 | 0.9042 |

## Test metrics (held-out test set)

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9303 | 0.9938 | 0.9305 | 0.9303 | 0.9303 | 0.9157 |
| Decision Tree | 0.9126 | 0.9798 | 0.9127 | 0.9126 | 0.9123 | 0.8943 |
| kNN | 0.9282 | 0.9912 | 0.9291 | 0.9282 | 0.9283 | 0.9132 |
| Naive Bayes | 0.9022 | 0.9909 | 0.9038 | 0.9022 | 0.9025 | 0.8822 |
| Random Forest (Ensemble) | 0.9318 | 0.9934 | 0.9321 | 0.9318 | 0.9317 | 0.9174 |

**Overall winner (highest test F1): Random Forest (Ensemble)**
