# The 14-Stage Scientific Research Lifecycle

AirIntel is structured as a reproducible research journey spanning **14 structured Jupyter notebooks** in `notebooks/`.

---

## Lifecycle Overview

```text
01. Ingestion ──► 02. Profiling ──► 03. Feature Factory ──► 04. Feature Selection
                                                                     │
┌────────────────────────────────────────────────────────────────────┘
▼
05. Model Benchmarks ──► 06. Optuna Tuning ──► 07. TreeSHAP ──► 08. Spatial Moran
                                                                     │
┌────────────────────────────────────────────────────────────────────┘
▼
09. Serialization ──► 10. Serving Engine ──► 11. Telemetry Logs
                                                   │
┌──────────────────────────────────────────────────┘
▼
12. Drift Audit (PSI) ──► 13. Governance ──► 14. Production MLOps
```

---

## Notebook Stages at a Glance

### Stage 01: Multi-Station Telemetry Ingestion & Audit
* **Notebook**: `notebooks/01_Data_Extraction.ipynb`
* **Objective**: Ingest, harmonize, and audit continuous CPCB CAAQMS sensor telemetry across 29 urban monitoring centers.
* **Key Finding**: Identified >35% missingness; established validation schema for particulate mass and gaseous channels.

### Stage 02: Exploratory Atmospheric Profiling & Seasonal Analysis
* **Notebook**: `notebooks/02_EDA_Atmospheric_Profiling.ipynb`
* **Objective**: Characterize microclimatic variations, seasonal cycles, and spatial cross-city correlations.
* **Key Finding**: Quantified the 3.2× winter particulate surge in the Indo-Gangetic Basin and 75% monsoon rain aerosol scavenging in coastal hubs.

<div class="doc-image-card">
  <img src="../assets/images/city_ranking.png" alt="National Urban Center PM2.5 Rankings" style="max-height: 440px; width: auto;" />
  <div class="doc-image-caption">Figure: Multi-City Baseline Particulate Concentrations Across 29 Indian Urban Monitoring Centers</div>
</div>

### Stage 03: The Atmospheric Feature Engineering Factory
* **Notebook**: `notebooks/03_Feature_Engineering.ipynb`
* **Objective**: Synthesize 233 candidate features across temporal harmonics, thermodynamic interactions, and geographical boundaries.
* **Key Finding**: Cyclic sin/cos harmonic features eliminated artificial boundary discontinuities at midnight and year-end transitions.

### Stage 04: 8-Way Consensus Feature Selection
* **Notebook**: `notebooks/04_Consensus_Feature_Selection.ipynb`
* **Objective**: Prune 233 features down to a lean 36-feature set using multi-algorithmic consensus.
* **Key Finding**: Pruning removed collinear redundancies while preserving 98.4% of predictive signal, cutting inference latency to < 20 ms.

### Stage 05: Model Zoo Benchmarking
* **Notebook**: `notebooks/05_Model_Zoo_Benchmarking.ipynb`
* **Objective**: Benchmark Linear, Ridge, Random Forest, XGBoost, CatBoost, and LightGBM models.
* **Key Finding**: LightGBM achieved champion continuous regression performance (R² = 0.8874), while CatBoost led 6-class severity classification (89.4%).

### Stage 06: Optuna Bayesian Hyperparameter Optimization
* **Notebook**: `notebooks/06_Optuna_Bayesian_Optimization.ipynb`
* **Objective**: 100-trial Bayesian search using Tree-structured Parzen Estimators (TPE) with 5-fold cross-validation.
* **Key Finding**: Delivered a 40.4% RMSE reduction on LightGBM and calibrated CatBoost class probabilities.

### Stage 07: TreeSHAP Game-Theoretic Explainability
* **Notebook**: `notebooks/07_TreeSHAP_Explainability.ipynb`
* **Objective**: Compute exact Shapley attributions for live decision explainability.
* **Key Finding**: Decomposed predictions into additive pushes from nationwide baseline E[f(x)] = 112.5 AQI.

### Stage 08: Spatial Autocorrelation & Geographic Clustering
* **Notebook**: `notebooks/08_Spatial_Analysis_Moran.ipynb`
* **Objective**: Measure spatial autocorrelation and geographic clustering of pollution and residuals.
* **Key Finding**: Global Moran's I = 0.412 (p < 0.001), confirming significant spatial clustering of winter particulate stagnation.

### Stage 09: Model Artifact Packaging & Bundle Serialization
* **Notebook**: `notebooks/09_Model_Packaging_Serialization.ipynb`
* **Objective**: Package estimators, encoders, medians, and coordinates into a self-contained 3.57 MB pickle asset.
* **Key Finding**: Standalone deployment bundle (`deployment_pipeline.pkl`) enables instant loading with zero cloud dependencies.

### Stage 10: Interactive Dual-Architecture Serving Centerpiece
* **Notebook**: `notebooks/10_Interactive_Prediction_Engine.ipynb`
* **Objective**: Deliver interactive dual-mode prediction (Scientific Full Chemical Array vs Public Weather-Driven Option A).
* **Key Finding**: Integrated live inference, interactive sliders, gauge meter, and dynamic precaution directives into Streamlit.

### Stage 11: Real-Time Inference Telemetry Logging
* **Notebook**: `notebooks/11_Inference_Telemetry_Logging.ipynb`
* **Objective**: Implement asynchronous, non-blocking telemetry logging into structured JSONL.
* **Key Finding**: Automatically tracks timestamp, city, mode, predicted AQI, EPA severity tier, and model confidence score.

### Stage 12: Statistical Drift Detection (PSI & KS-Test)
* **Notebook**: `notebooks/12_Statistical_Drift_Monitoring.ipynb`
* **Objective**: Detect population distribution shifts and concept drift in live production inference.
* **Key Finding**: Automated Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov test alert protocols.

### Stage 13: Operational Governance & Retraining Cadence
* **Notebook**: `notebooks/13_Retraining_Governance.ipynb`
* **Objective**: Codify operational runbooks, drift threshold triggers (PSI ≥ 0.25), and model rollback procedures.
* **Key Finding**: Established strict retraining criteria, avoiding unnecessary model churn while protecting against concept degradation.

### Stage 14: Production MLOps, CI/CD & Automated Testing
* **Notebook**: `notebooks/14_MLOps_Reproducibility_Monitoring.ipynb`
* **Objective**: End-to-end MLOps pipeline covering DVC data tracking, MLflow experiments, 5-point Pytest suite, and Docker containerization.
* **Key Finding**: Full verification suite ensures complete reproducibility and enterprise production standards.
