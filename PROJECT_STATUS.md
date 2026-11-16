# SoftwarePulse - Project Status

## Implemented & Validated Modules

| Module | Status | Verification |
| :--- | :---: | :--- |
| **Repository Ingestion** | ✓ | Local git repositories & remote clone support |
| **Git History Extraction** | ✓ | Commit history, churn tracking, author analytics |
| **AST Source-Code Metrics** | ✓ | LOC, McCabe Cyclomatic Complexity, functions, classes, imports (`analyze_source_content` in-memory) |
| **Dependency Graph & Architecture** | ✓ | AST import resolver & NetworkX DiGraph structural metrics (`build_graph_from_metrics`) |
| **Historical Snapshot Reconstruction** | ✓ | Read-only tree extraction via `SnapshotReconstructor` (`git ls-tree`, `git show`) without checking out commits |
| **Temporal Dataset Construction** | ✓ | Strict chronological observation windows: $X(T)$ historical, $y(T)$ bounded future horizon $(T, T+H]$ |
| **Temporal Leakage Prevention** | ✓ | Dual independent train/test future horizons; boundary assertions (`validate_temporal_boundaries`) |
| **Target Label Generation & Provenance** | ✓ | Configurable bugfix keyword heuristics, contributing commit tracking, explicit proxy fallback, no fake labels |
| **Repository Health Score (Feature 1)** | ✓ | 5-dimension engine (Code Quality 25%, Change Risk 25%, Dependency Health 20%, Stability 20%, Test Health 10%), grades A–F, weakness detection |
| **Intelligent Recommendations (Feature 2)** | ✓ | Rule-driven, evidence-backed prioritized guidance (CRITICAL, HIGH, MEDIUM, LOW) across complexity, churn, cycles, hubs, bus factor, and tests |
| **Interactive Graph & Risk Propagation (Feature 3)** | ✓ | Plotly network diagrams with 3-level hierarchy, cycle detection, risk cascade simulation (decay factor), and architecture diff |
| **Pull Request Risk Analyzer (Feature 4)** | ✓ | Local branch (`base..head`) and GitHub PR evaluation, AST diff parsing, ML risk delta prediction, and downstream exposure discovery |
| **Feature Ablation Experiments (Feature 5)** | ✓ | 4-way ablation (Static, Static+Evolution, Static+Dependency, Full) vs Majority Baseline, honest F1 & ROC-AUC reporting |
| **Machine Learning Models** | ✓ | Random Forest & Logistic Regression classifiers with pipelines fitted strictly on training data |
| **Explainable AI** | ✓ | Global feature importances & localized component risk factors |
| **What-If Simulation Engine** | ✓ | Dependency mutations, complexity shifts, and risk delta recalculation |
| **Developer UI & Dashboard** | ✓ | Professional dark-theme UI with 15 dedicated pages, responsive sidebar, breadcrumbs, and metrics |
| **Comprehensive Test Suite** | ✓ | 88 passed tests across all units, analyzers, temporal boundaries, ablation models, health scores, and PR analyzer |

---

## Planned Future Work (Post-Midsem / Final Semester Envisioned)

| Module | Status |
| :--- | :---: |
| Temporal Graph Neural Network (TGNN) | ○ |
| Full Temporal Knowledge Graph | ○ |
| Technical Debt Forecasting | ○ |
| Architecture Degradation Forecasting | ○ |
| Advanced Causal Inference (Do-Calculus / SCM) | ○ |
| Developer Collaboration Social Graph | ○ |
| Multi-language Polyglot Parsers (Tree-sitter) | ○ |
| Continuous CI/CD Bot Integration | ○ |
