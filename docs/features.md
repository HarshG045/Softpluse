# SoftwarePulse - Features & Capabilities

SoftwarePulse is an evolution-aware software risk intelligence platform that models how codebase complexity, temporal churn, and architectural dependency topology interact to predict component-level risk.

---

## 1. Repository Health Score (Feature 1)

The **Repository Health Score** provides an empirical 0–100 rating and letter grade (A through F) evaluating overall repository condition.

### Health Dimensions & Scoring Formulation

$$Score_{overall} = \sum_{i=1}^{5} w_i \cdot Score_i$$

| Dimension | Weight ($w_i$) | Primary Indicators | Penalties & Benchmarks |
| :--- | :---: | :--- | :--- |
| **Code Quality** | 25% | Cyclomatic Complexity, LOC, function/class counts | Penalized by files with $CC > 15$ (-35 pts max) and $LOC > 500$ (-20 pts max). |
| **Change Risk** | 25% | Git churn volume, commit frequency, author distribution | Penalized by high churn (>1,000 lines, -25 pts max) and single-author bus factor (-20 pts max). |
| **Dependency Health** | 20% | Directed graph coupling, circular dependency cycles, hub centrality | Penalized heavily by circular cycles (-25 pts base) and coupling hubs ($degree > 10$). |
| **Stability** | 20% | Historical bug-fix commits, churn concentration | Derived from defect ratio and top 20% churn concentration. |
| **Test Health** | 10% | Test file presence, test-to-source file ratio ($R = N_{test} / N_{src}$) | Benchmarked against target ratio of 0.50. *Does not claim test execution coverage.* |

### Letter Grade Classification
- **Grade A (85–100):** Healthy architecture, modular boundaries, manageable churn, low defect exposure.
- **Grade B (70–84):** Acceptable quality; isolated complexity hotspots or minor churn concentration.
- **Grade C (55–69):** Moderate architectural debt; elevated coupling or cognitive complexity in core modules.
- **Grade D (40–54):** High maintenance burden; circular dependencies, high churn, or bus factor risk.
- **Grade F (< 40):** Critical structural degradation; pervasive circular cycles, extreme complexity, absence of tests.

---

## 2. Intelligent Risk Recommendations (Feature 2)

SoftwarePulse features an automated, rule-driven recommendation engine that translates quantitative metrics and ML risk scores into prioritized engineering actions.

### Severity Tiers
1. **CRITICAL:** Architectural circular dependencies or extreme complexity hotspots requiring immediate refactoring.
2. **HIGH:** High-risk components under active churn, single-author ownership risk, or coupling hubs.
3. **MEDIUM:** Moderate complexity growth, undocumented public modules, or coupling imbalances.
4. **LOW:** Test suite expansion targets, file documentation, and cleanups.

### Implemented Rule Catalog
- `RULE_CIRCULAR_DEPENDENCY`: Identifies strongly connected components and directed cycles; provides decoupling guidance.
- `RULE_EXTREME_COMPLEXITY`: Triggers when McCabe Cyclomatic Complexity exceeds 20.
- `RULE_HIGH_RISK_UNDER_CHURN`: Triggers when a component has high predicted risk ($\ge 60\%$) and active churn.
- `RULE_DEPENDENCY_COUPLING_HUB`: Triggers when component degree centrality exceeds 10.
- `RULE_SINGLE_AUTHOR_BUS_FACTOR`: Identifies high-risk components with only 1 author across $\ge 5$ commits.
- `RULE_MISSING_TEST_COMPANION`: Identifies business logic without an associated `test_*.py` file.
- `RULE_HIGH_CHANGE_FREQUENCY`: Flags files modified in more than 30% of repository commits.

---

## 3. Interactive Dependency Graph & Risk Propagation (Feature 3)

The architecture explorer models the repository as a directed graph $G = (V, E)$, where vertices represent Python components and edges represent imports.

### Interactive Modes
- **Architecture Overview:** High-level package cluster topology showing inter-package afferent/efferent coupling and instability indices.
- **Component Explorer:** 3-level directed neighborhood view around a selected component, supporting Hierarchical, Radial, and Grouped layouts.
- **Risk Propagation (Blast Radius):** Simulates how risk originating at a critical component cascades through downstream dependents:
  $$Risk_{propagated}(v) = Risk_{origin} \cdot \delta^{depth(v)}$$
  where $\delta \in [0.1, 0.95]$ is a configurable decay factor.
- **Architecture Diff:** Visualizes structural topology diffs, highlighting added, deleted, and modified dependency edges.

---

## 4. Pull Request Risk Analyzer (Feature 4)

Pre-merge risk forecasting evaluates proposed changes before code is merged into the default branch.

### Capabilities
- **Local Branch Comparison (`base..head`):** Evaluates Git diffs directly between branches without checking out code.
- **Remote GitHub Pull Request:** Connects via read-only GitHub API, parses unified diffs, and computes structural shifts.
- **Structural Shift Prediction:** Reconstructs AST complexity delta, LOC changes, and dependency modifications from patch text.
- **Downstream Blast Radius:** Traces which dependent modules in the dependency graph are affected by the PR.
- **Automated Verification Suggestions:** Recommends targeted regression tests based on impacted components.

---

## 5. Feature Ablation Research Experiments (Feature 5)

SoftwarePulse addresses the fundamental research question:
> *"Does incorporating software evolution and dependency topology improve component-level risk prediction compared with static metrics alone?"*

### Compared Model Configurations
1. **Baseline Model:** Majority-class dummy classifier ($Zero-R$).
2. **Model A (Static Only):** 6 AST features (`loc`, `complexity`, `functions`, `classes`, `imports`, `methods`).
3. **Model B (Static + Evolution):** 13 features adding Git churn, commit frequency, and author counts.
4. **Model C (Static + Dependency):** 12 features adding degree, centrality, in-degree, and out-degree.
5. **Model D (Full Model):** 19 features combining static, evolution, and graph topology.

### Leakage-Free Scientific Protocol
- **Strict Chronological Windows:** $X(T)$ is computed strictly at observation cutoff $T_{obs}$; $y(T)$ is evaluated over bounded future horizon $(T_{obs}, T_{obs} + H]$.
- **Independent Train and Test Horizons:** Training label horizon is non-overlapping with testing observation window.
- **Safe Read-Only Snapshots:** Tree-based Git extraction (`git ls-tree`, `git show`) prevents working tree corruption.
- **Fair Pipeline Scaling:** Scaler and imputers are fitted strictly on training data.
