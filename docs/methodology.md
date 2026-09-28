# SoftwarePulse Research Methodology

## 1. Primary Research Question

> **"Does incorporating software evolution and dependency information improve component-level risk prediction compared with using static source-code characteristics alone?"**

## 2. Experimental Setup & Feature Sets

To evaluate this hypothesis rigorously, three model configurations are compared under identical training and testing splits:

| Model Configuration | Feature Groups Included | Total Features |
| :--- | :--- | :--- |
| **Model A: Static Only** | LOC, Cyclomatic Complexity, Functions, Classes, Imports, Methods | 6 |
| **Model B: Static + Evolution** | Model A + Commit Count, Recent Commits, Churn, Recent Churn, Authors, Change Frequency, Days Since Last Change | 13 |
| **Model C: Full (Evolution + Dependency)** | Model B + Dependency Count, Dependent Count, Total Degree, Centrality, In-Degree, Out-Degree | 19 |

## 3. Data Leakage Prevention (Temporal Splitting)

Traditional random k-fold cross-validation introduces severe temporal data leakage in software evolution analysis, as future code metrics and churn behavior would leak into the training set.

SoftwarePulse enforces strict **chronological splitting**:
- Commits up to time cutoff $T_{split}$ (70% of chronological timeline) are used to extract training features.
- Commits from $T_{split}$ to $T_{split} + W$ provide ground-truth target labels.
- Evaluation features represent subsequent state, evaluated strictly on unseen future behavior.

## 4. Ground-Truth Risk Labeling

Target labels are formulated based on whether a component is modified by a future bug-fixing commit:
- Bug-fixing commits are detected using keyword heuristics (`fix`, `bug`, `defect`, `patch`, `error`, `crash`, `issue`, etc.) in commit messages.
- If bugfix commits are unavailable or sparse in small repositories, the system automatically falls back to an upper-percentile future churn volatility proxy, explicitly documented in the UI.

## 5. Machine Learning Classifiers & Evaluation Metrics

- **Classifier**: Random Forest Classifier with `class_weight='balanced'` and standard depth regularization.
- **Baseline**: Logistic Regression with standard scaling pipeline.
- **Metrics**: Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix (True Positives, False Positives, True Negatives, False Negatives).
