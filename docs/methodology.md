# SoftwarePulse Research Methodology

## 1. Primary Research Question

> **"Does incorporating software evolution and dependency information improve component-level risk prediction compared with using static source-code characteristics alone?"**

## 2. Experimental Setup & Feature Sets

To evaluate this hypothesis rigorously, three model configurations are compared under identical training and testing samples and target labels:

| Model Configuration | Feature Groups Included | Total Features |
| :--- | :--- | :--- |
| **Model A: Static Only** | LOC, Cyclomatic Complexity, Functions, Classes, Imports, Methods | 6 |
| **Model B: Static + Evolution** | Model A + Commit Count, Recent Commits, Churn, Recent Churn, Authors, Change Frequency, Days Since Last Change | 13 |
| **Model C: Full (Evolution + Dependency)** | Model B + Dependency Count, Dependent Count, Total Degree, Centrality, In-Degree, Out-Degree | 19 |

## 3. Strict Chronological Prediction Task

To eliminate data leakage, SoftwarePulse formulates a strictly chronological prediction task:

For each component $c$:
- **Observation timestamp $T$**: The point in history at which predictions are made.
- **Historical feature window $(\dots, T]$**: Only information available at or before $T$ is used.
  - $X_{static}(T)$: AST metrics parsed from source files as they existed at commit $C(T)$ via `git show`.
  - $X_{evolution}(T)$: Git churn, commits, and authors computed using only commits up to $T$.
  - $X_{dependency}(T)$: Intra-repository dependency graph topology reconstructed from imports at $C(T)$.
- **Future label horizon $(T, T + H]$**: Target label $y(T)$ indicates whether component $c$ experiences a qualifying defect-related event within future horizon $H$.

```
Timeline:
|--- Train History ---|--- Train Labels (H) ---|--- Test History ---|--- Test Labels (H) ---|
                      T_train_obs              T_train_lbl_end      T_test_obs              T_test_lbl_end
```

### Dual Independent Horizons

1. **Training Partition**:
   - Features $X_{train}$ extracted as of $T_{train\_obs}$.
   - Labels $y_{train}$ observed strictly within $(T_{train\_obs}, T_{train\_lbl\_end}]$.
2. **Testing Partition**:
   - Features $X_{test}$ extracted as of $T_{test\_obs}$ (where $T_{test\_obs} \ge T_{train\_lbl\_end}$).
   - Labels $y_{test}$ observed strictly within $(T_{test\_obs}, T_{test\_lbl\_end}]$.

**Critical Guarantees:**
- No future commit data enters historical features.
- No present-day source code or dependency topology is used in historical observations.
- Training and testing label windows are disjoint and non-overlapping.
- Training labels never depend on commits after the training evaluation horizon.

## 4. Snapshot-Based Historical Feature Reconstruction

Feature extraction at historical commits is performed safely without modifying the working directory or checking out commits:
- `SnapshotReconstructor` uses read-only `git ls-tree` to enumerate files present at commit $C(T)$ and `git show <commit>:<path>` to retrieve file contents.
- `CodeAnalyzer.analyze_source_content` parses AST metrics in-memory.
- `DependencyAnalyzer.build_graph_from_metrics` reconstructs the directed dependency graph from historical import statements.
- File identities, additions, renames, and deletions across commits are naturally preserved according to the Git tree at that commit.

## 5. Ground-Truth Risk Labeling & Provenance Tracking

Target labels are derived using:
1. **Configurable Bug-Fix Heuristic (Primary)**:
   - Identifies bug-fixing commits matching configurable keywords (`fix`, `bug`, `defect`, `patch`, `error`, `crash`, `issue`, `resolve`, `close`, `regression`, `fault`, `vulnerability`, `leak`).
   - Components modified in qualifying bug-fixing commits receive label `y = 1`, others `0`.
   - Every positive label records its contributing commit hashes in `LabelSummary.component_provenance`.
2. **High-Churn Volatility Proxy (Documented Fallback)**:
   - If bug-fix data is absent or sparse (single class), the system can fall back to an upper-percentile future churn proxy label.
   - When active, the label is explicitly flagged as `is_proxy = True` in dataset metadata and prominently badge-labeled in the UI. Proxy labels and defect labels are never silently mixed.
3. **Insufficient Data Handling**:
   - If a repository has no qualifying events in the future window and proxy labels are disabled, the system reports `INSUFFICIENT_DATA` honestly without fabricating synthetic labels.

## 6. Machine Learning Classifiers & Honest Evaluation

- **Classifiers**: Random Forest Classifier (`class_weight='balanced'`) and Logistic Regression (`StandardScaler` fitted strictly on training data).
- **Baseline**: Majority-class dummy classifier provided for experimental context.
- **Fair Model Comparison**: All three models (A, B, C) are evaluated on identical test samples and identical target labels.
- **Evaluation Metrics**:
  - Precision, Recall, F1-Score, Accuracy, Confusion Matrix.
  - **ROC-AUC**: Computed only when the holdout test set contains both classes ($y \in \{0, 1\}$). For single-class holdout sets, ROC-AUC is reported as `N/A` rather than a fabricated or misleading score.
- **Validation Checks**:
  - `TemporalSplitter.validate_temporal_boundaries` raises `TemporalLeakageError` on overlapping label windows or inverted observation times.
  - `TemporalSplitter.validate_no_target_in_features` asserts no ground-truth columns exist in feature matrices.

## 7. Known Limitations

- **Heuristic Bug Fix Identification**: Commit messages may not describe all fixes or may misclassify refactorings as fixes.
- **Short Repositories**: Repositories with fewer than 4 commits cannot support two independent non-overlapping future horizons; these are documented with clear insufficiency warnings.
- **Dynamic Imports**: AST analysis captures static import statements; dynamic `__import__` or runtime plugin loading is not reflected in the static dependency graph.
