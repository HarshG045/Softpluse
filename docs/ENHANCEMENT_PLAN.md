# SoftwarePulse Enhancement Plan & Architecture Specification

## 1. Executive Summary

This document specifies the architectural enhancement plan for **SoftwarePulse**, advancing it into a research-backed, developer-focused software risk intelligence platform.

The plan builds directly upon the existing codebase without rewriting working components, preserves 100% of existing functionality, and implements the five requested capabilities:
1. **Repository Health Score** (5-dimensional 0–100 quality, stability, change, dependency, and test health evaluation)
2. **Intelligent Risk Recommendations** (Evidence-backed, rule-driven action items with supporting metrics)
3. **Interactive Dependency Graph & Risk Propagation** (NetworkX + Plotly hierarchical visualizer, neighborhood inspection, transitive impact, cycle alerts, and historical architectural diffs)
4. **Pull Request Risk Analyzer** (Local branch & GitHub PR change analysis, diff parsing, complexity/dependency delta evaluation, and pre-merge risk scoring)
5. **Feature Ablation Research Suite** (Rigorous 4-way feature ablation across Static, Static+Evolution, Static+Dependency, and Full configurations with majority baseline and leak-free temporal evaluation)
6. **Unified Developer UI/UX Redesign** (Categorized navigation shell, responsive layout, health dashboards, component drill-downs, and rich visualization styling)

---

## 2. Existing Architecture & Reusable Components

The project currently has **77 passing tests** with 0 regressions. Reusable components include:

| Subsystem | Existing Modules | Reusability in Enhancement |
|:---|:---|:---|
| **Ingestion** | `repository_loader.py`, `git_loader.py`, `repository_metadata.py` | Full reuse for local/remote repo loading, commit logs, branch comparisons, and diff parsing. |
| **Analysis** | `code_analyzer.py`, `git_analyzer.py`, `dependency_analyzer.py`, `snapshot_reconstructor.py`, `change_analyzer.py`, `historical_analyzer.py`, `snapshot_engine.py` | Full reuse. In-memory AST metric extraction, DiGraph construction, and snapshot reconstruction form the foundation for PR analysis and architecture diffs. |
| **Features** | `feature_schema.py`, `feature_engineering.py` | Add `STATIC_PLUS_DEPENDENCY` schema (12 features) to complement existing `STATIC`, `STATIC_PLUS_EVOLUTION`, and `FULL`. |
| **Dataset** | `builder.py`, `labeling.py`, `temporal_split.py` | Full reuse of dual-horizon temporal splitting, bounded label horizons, and provenance tracking. |
| **ML & Evaluation** | `train.py`, `predict.py`, `evaluate.py`, `model_manager.py` | Full reuse. Pipelines with imputer/scaler, Random Forest, Logistic Regression, honest ROC-AUC, and single-class handling. |
| **Explainability** | `explainer.py` | Full reuse for signal attribution, feeding directly into recommendation generation. |
| **Simulation** | `what_if.py` | Full reuse of counterfactual graph mutation and risk sensitivity recalculation. |
| **Visualization** | `dependency_graph.py`, `risk_charts.py`, `evolution_charts.py`, `component_view.py` | Upgrade Plotly layouts, integrate new health gauges, ablation charts, and PR diff visualizers. |
| **UI** | `sidebar.py`, `header.py`, `pages.py`, `styles.py`, `components.py` | Reorganize into modern hierarchical navigation sections without breaking existing page routes. |

---

## 3. New Modules to Create

```
SoftPluse/
├── health/
│   ├── __init__.py
│   ├── health_metrics.py         # Dimension metrics extraction & normalization
│   └── health_score.py           # Multi-dimensional repository health scoring engine
├── recommendations/
│   ├── __init__.py
│   ├── recommendation_rules.py   # Rule definitions, thresholds, priorities
│   └── recommendation_engine.py  # Evidence-based recommendation generator
├── pull_requests/
│   ├── __init__.py
│   ├── github_client.py          # Read-only GitHub API integration (token-safe)
│   ├── pr_diff_parser.py         # Unified diff parser for git diff / PR patches
│   └── pr_risk_analyzer.py       # Pre-merge risk predictor & impact evaluator
├── experiments/
│   ├── __init__.py
│   └── ablation.py               # 4-way feature ablation runner & comparative reporting
├── tests/
│   ├── test_health_score.py      # Health score calculation & dimension tests
│   ├── test_recommendations.py   # Rule triggering & provenance tests
│   ├── test_pr_analyzer.py       # Branch comparison & PR risk tests
│   └── test_ablation.py          # 4-way ablation & baseline tests
```

---

## 4. Detailed Specification by Feature

### Feature 1: Repository Health Score (`health/`)
- **Dimensions**:
  1. *Code Quality (Weight: 25%)*: Cyclomatic complexity distribution, median LOC, function density, maintainability index proxy.
  2. *Change Risk (Weight: 25%)*: Churn volatility, recent commit frequency, contributor spread, days since last change.
  3. *Dependency Health (Weight: 20%)*: Average coupling, presence of circular dependency cycles, density of architectural hubs/bridges.
  4. *Stability (Weight: 20%)*: Defect recurrence (bugfix commit density), volatility of modified components across historical snapshots.
  5. *Test Health (Weight: 10%)*: Ratio of test files (`test_*.py`, `*_test.py`) to implementation files, dedicated test directories. Transparently notes when line coverage data is not available (never claims false coverage).
- **Output**: Overall score (0–100), grade (A/B/C/D/F), 5 dimension scores with full metric breakdowns, top repository weaknesses, and concrete improvement targets.

### Feature 2: Intelligent Risk Recommendations (`recommendations/`)
- **Rule Engine**:
  - `HighComplexityRule`: Triggers when McCabe complexity $\ge 10$ or branches $\ge 8$. Action: modular decomposition and unit branch testing.
  - `HighChurnVolatilityRule`: Triggers when recent churn is in top 20th percentile. Action: code review, regression testing.
  - `ArchitecturalHubRule`: Triggers when component has degree $\ge 4$ or high centrality. Action: contract testing, dependency decoupling.
  - `CircularDependencyRule`: Triggers when component participates in an import cycle. Action: interface segregation, dependency inversion.
  - `SingleAuthorExposureRule`: Triggers when a component has high churn but only 1 contributor. Action: pair programming, knowledge transfer.
  - `LowTestAffinityRule`: Triggers when a high-risk component lacks matching test files. Action: add targeted unit test suite.
- **Output**: Prioritized list (CRITICAL, HIGH, MEDIUM, LOW) containing recommended action, specific reason, supporting numerical metrics, and suggested developer next step.

### Feature 3: Interactive Dependency Graph & Risk Propagation (`visualization/`)
- **Interactive Graph Explorer**:
  - Full support for Plotly pan, zoom, node selection, and neighborhood highlighting.
  - Directional filtering (upstream dependencies, downstream dependents, or both).
  - Package/directory grouping, degree filters, and risk level filtering.
  - Visual distinction: green (LOW), amber (MEDIUM), red (HIGH), with accessible text badges `[L]`, `[M]`, `[H]`.
- **Circular Dependency Alerts**: Highlights cycles in bright red with dashed links and tooltip diagnostic paths.
- **Experimental Risk Propagation**:
  - Formula: $\text{Impact}(c) = \text{Risk}(c) \times \sum_{d \in \text{Dependents}(c)} \frac{1}{\text{depth}(d) + 1}$
  - Explicit scientific disclaimer: *Estimates potential architectural blast radius based on dependency depth; does not claim direct component failure.*
- **Historical Architecture Comparison**:
  - Compares dependency graphs across two historical commits using `SnapshotReconstructor`.
  - Highlights added edges, removed edges, centrality shifts, and newly introduced cycles.

### Feature 4: Pull Request Risk Analyzer (`pull_requests/`)
- **Inputs**:
  - Local branch comparison (e.g., `main..feature_branch`).
  - GitHub PR URL (e.g., `https://github.com/org/repo/pull/123`) using anonymous public API or user-configured token.
- **Pipeline**:
  1. Extract changed files via `GitLoader.compare_branches` or GitHub API patch.
  2. Parse file-level additions, deletions, and structural keyword shifts (branches, imports).
  3. Estimate complexity and dependency shift using `ChangeImpactAnalyzer`.
  4. Run modified feature representation through trained ML pipeline to predict risk delta ($\Delta$).
  5. Identify affected downstream components via `DependencyGraph`.
  6. Generate structured PR risk report with risk tier (LOW / MEDIUM / HIGH), key risk factors, and recommended verification checklist.

### Feature 5: Feature Ablation & Research Suite (`experiments/`)
- **4 Research Configurations**:
  1. *Model A*: Static Only (6 features: `loc`, `complexity`, `functions`, `classes`, `imports`, `methods`)
  2. *Model B*: Static + Evolution (13 features: Model A + 7 Git metrics)
  3. *Model C*: Static + Dependency (12 features: Model A + 6 structural metrics)
  4. *Model D*: Full (19 features: Model A + Evolution + Dependency)
  5. *Baseline*: Majority-class dummy classifier for statistical context.
- **Integrity**:
  - All models evaluated on identical sample instances, identical temporal split, and identical ground-truth targets.
  - Scaling fitted strictly on training data; no future data leakage.
  - Honest single-class reporting (no fake labels, ROC-AUC reported as N/A when mathematically undefined).
  - Feature contribution delta table ($F_1$ gain over static baseline).

### Feature 6: UI/UX Redesign (`ui/`)
- **Navigation Hierarchy**:
  - `OVERVIEW`: Dashboard (Health Score, key metrics, risk distribution, trends), Repository Explorer
  - `CODE INTELLIGENCE`: Risk Explorer, Component Details, Interactive Dependency Graph, Architecture Overview
  - `EVOLUTION`: Git Activity, Evolution Timeline, Architecture Evolution
  - `PREDICTION & SIMULATION`: Risk Prediction, Explainability, What-If Lab, Recommendations
  - `RESEARCH & EVALUATION`: Model Evaluation, Feature Ablation, Experiment Details
  - `INTEGRATIONS`: Pull Request Analyzer, GitHub Review
  - `SETTINGS`: Configuration, Thresholds, CSV/JSON Exports
- **Styling**: Developer-focused GitHub Dark theme (`#0d1117`, `#161b22`, `#21262d`, `#58a6ff`), responsive metrics cards, clean Plotly charts with Inter/JetBrains Mono typography.

---

## 5. Implementation Roadmap & Order of Work

```
Phase 0: Project Audit & Plan [CURRENT - COMPLETE]
  └── docs/ENHANCEMENT_PLAN.md created.

Phase 1: Feature Ablation & Research Experiments
  ├── Update features/feature_schema.py (add STATIC_PLUS_DEPENDENCY)
  ├── Create experiments/ablation.py
  ├── Update ml/evaluate.py to support 4-way ablation
  └── Add tests/test_ablation.py

Phase 2: Repository Health Score & Recommendations
  ├── Create health/health_metrics.py & health/health_score.py
  ├── Create recommendations/recommendation_rules.py & recommendations/recommendation_engine.py
  ├── Add tests/test_health_score.py
  └── Add tests/test_recommendations.py

Phase 3: Interactive Architecture & Risk Propagation
  ├── Enhance visualization/dependency_graph.py (risk propagation & architecture diff)
  └── Update ui/pages.py (Dependency Explorer & Architecture pages)

Phase 4: Pull Request Risk Analyzer
  ├── Create pull_requests/pr_diff_parser.py
  ├── Create pull_requests/github_client.py
  ├── Create pull_requests/pr_risk_analyzer.py
  └── Add tests/test_pr_analyzer.py

Phase 5: Unified UI/UX Redesign & Navigation Shell
  ├── Update ui/sidebar.py with categorized navigation
  ├── Update ui/header.py & ui/components.py with health badges & recommendation cards
  ├── Update ui/pages.py (Overview, Health, PR Analyzer, Ablation pages)
  └── Update app.py workflow orchestration

Phase 6: Verification, Documentation & Final Report
  ├── Run full test suite (aiming for 100% pass rate)
  ├── Update docs/methodology.md, docs/architecture.md, docs/user_guide.md, README.md, PROJECT_STATUS.md
  └── Compile final enhancement report
```

---

## 6. Risk Assessment & Mitigations

| Risk | Mitigation |
|:---|:---|
| Network dependency during GitHub PR inspection | Provide robust local Git branch comparison as default offline mode; GitHub API is strictly opt-in with graceful error handling when unauthenticated. |
| Performance degradation on large dependency graphs | Implement depth filters (depth 1–3), package-level clustering view, and node cap thresholds (max 60 nodes in detail view). |
| Mathematical validity of ablation comparison | Enforce that all 4 models evaluate on the exact same holdout split generated by `TemporalSplitter`. |
| UI state management in Streamlit | Store analyzed results in `st.session_state.analysis_data` with clean caching to prevent re-running analysis on tab switches. |
