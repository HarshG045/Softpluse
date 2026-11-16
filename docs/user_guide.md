# SoftwarePulse - User Guide

A step-by-step developer and researcher guide for running, exploring, and evaluating repositories with SoftwarePulse.

---

## 1. Quickstart & Launching

### Installation
Ensure Python 3.11+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/HarshG045/Softpluse.git
cd Softpluse
pip install -r requirements.txt
```

### Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 2. Analyzing a Repository

From the sidebar:
1. **Choose Source:**
   - **Demo Microservices Repo:** Generates a synthetic multi-commit microservices repository with authentic Python code, dependencies, and commit history.
   - **Local Git Repository:** Enter the absolute or relative path to any local Git repository (e.g., `.` or `/path/to/my-repo`).
2. **Select Machine Learning Classifier:**
   - **Random Forest:** Non-linear decision tree ensemble with balanced class weights.
   - **Logistic Regression:** Linear classifier with L2 regularization and standard feature scaling.
3. **Click "Analyze Repository":**
   - The pipeline extracts AST metrics, builds dependency graphs, reconstructs historical snapshots, fits temporal models, computes health scores, and generates recommendations.

---

## 3. Navigating the Platform

The sidebar organizes features into four functional sections:

### Section A: Analysis
- **Overview:** Executive summary with health score banner, component risk breakdown, top prioritized components, and action recommendations.
- **Repository Health:** Deep-dive into the 5 health dimensions (Code Quality, Change Risk, Dependency Health, Stability, Test Health), weaknesses, and improvement targets.
- **Risk Explorer:** Detailed component-by-component inspection table, search autocomplete, category filters, and localized risk factor attributions.
- **Architecture:** Subsystem module clusters, internal/external coupling ratios, and architectural instability index ($I = \frac{C_e}{C_a + C_e}$).
- **Evolution:** Multi-snapshot trajectory plots showing how component complexity, LOC, and churn evolved over time.
- **Architecture Evolution:** Historical architecture comparison and structural shift metrics.
- **Git Activity:** Commit history timeline, author contribution distribution, and safe Git commit workflow.
- **Dependencies:** Interactive directed dependency graph with 4 viewing modes: Architecture Overview, Component Explorer, Risk Propagation (Blast Radius), and Architecture Diff.

### Section B: Intelligence
- **Action Recommendations:** Prioritized engineering actions (CRITICAL, HIGH, MEDIUM, LOW) with quantitative evidence, triggering metrics, and remediation guidance.
- **Pull Request Risk:** Pre-merge risk evaluation for local branch comparisons (`base..head`) and remote GitHub PR URLs.
- **Explainability:** Global SHAP/feature importance rankings and local attribution signals explaining why a component is high-risk.
- **What-If Lab:** Interactive architecture and complexity sandbox. Modify LOC, complexity, or add/remove dependencies to preview predicted risk shifts in real time.

### Section C: Research
- **Model Evaluation:** Detailed performance reports for Model A (Static), Model B (Static+Evolution), and Model C (Full), with confusion matrices, ROC-AUC, and single-class holdout transparency.
- **Feature Ablation:** 4-way ablation experiment directly answering whether evolution and graph topology improve defect risk prediction over static metrics alone.

### Section D: System
- **Settings:** Configurable risk thresholds, temporal prediction horizons ($H$), proxy labeling fallbacks, and export utilities.

---

## 4. Workflows & Common Use Cases

### Evaluating a Pull Request Before Merge
1. Open **Pull Request Risk** from the sidebar.
2. Under the **Local Branch Comparison** tab, select the base branch (`main`) and feature branch (`feature/my-branch`).
3. Click **Evaluate Branch Risk**.
4. Review the overall risk gauge, list of high-risk modified files, affected downstream dependents, and suggested unit/integration test suites.

### Simulating an Architectural Refactoring
1. Open **What-If Lab**.
2. Select the target component to refactor.
3. Adjust the proposed complexity reduction (e.g., reduce Cyclomatic Complexity by -10) or remove a tightly coupled dependency.
4. Observe the immediate predicted risk score delta (e.g., $72\% \rightarrow 38\%$, HIGH to LOW).

### Reviewing Feature Ablation Evidence
1. Open **Feature Ablation**.
2. Inspect the Comparative Model Evaluation Matrix across Baseline, Model A, Model B, Model C, and Model D.
3. Review the empirical delta synthesis showing the isolated quantitative contribution of evolution signals and dependency signals.
