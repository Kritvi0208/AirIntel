# MLOps, Drift Detection & Production Engineering

AirIntel includes a production-grade MLOps layer designed to ensure continuous reproducibility, statistical data drift monitoring, automated test verification, and containerized deployment.

---

## 1. Data Version Control (DVC) Architecture

Data lineage and intermediate transformations are codified in `dvc.yaml`:

```yaml
stages:
  clean_telemetry:
    cmd: python src/clean.py
    deps:
      - data/raw/
      - src/clean.py
    outs:
      - data/processed/clean_airintel.parquet

  feature_engineering:
    cmd: python src/features.py
    deps:
      - data/processed/clean_airintel.parquet
      - src/features.py
    outs:
      - data/processed/airintel_ml_final.parquet

  train_regression:
    cmd: python src/train_regression.py
    deps:
      - data/processed/airintel_ml_final.parquet
      - src/train_regression.py
    outs:
      - models/regressor/lightgbm_model.pkl

  train_classification:
    cmd: python src/train_classification.py
    deps:
      - data/processed/airintel_ml_final.parquet
      - src/train_classification.py
    outs:
      - models/classifier/catboost_model.pkl
```

---

## 2. MLflow Experiment Tracking

Experiments are logged to a lightweight SQLite backend (`sqlite:///mlflow.db`):
- **Experiment 1**: `AirIntel_AQI_Regression`
  - Parameters: `learning_rate=0.0382`, `num_leaves=63`, `n_estimators=850`, `max_depth=8`.
  - Metrics: R² = 0.8245, MAE = 12.46, RMSE = 16.54, Pearson r = 0.9084.
- **Experiment 2**: `AirIntel_Severity_Classification`
  - Parameters: `iterations=750`, `depth=7`, `learning_rate=0.045`, `loss_function=MultiClass`.
  - Metrics: Accuracy = 74.44%, Weighted F₁ = 0.7383, Macro F₁ = 0.5814.

---

## 3. Automated Pytest Verification Suite

The pipeline is verified by a 5-point production test suite (`tests/test_pipeline.py`):
1. **`test_data_integrity_and_bounds`**: Confirms target `US_AQI` strictly adheres to [0, 500], schema completeness, all 29 Indian cities present, and zero nulls.
2. **`test_pipeline_bundle_integrity`**: Validates deployment asset `deployment_pipeline.pkl` contains dual estimators, 36 selected features, valid cities, and feature medians.
3. **`test_single_row_inference_latency_sla`**: Benchmarks dual inference latency to meet production SLA (< 200 ms on CPU).
4. **`test_classification_probability_distribution`**: Validates that CatBoost predicted class probabilities sum strictly to 1.0 ± 1e-4.
5. **`test_feature_consistency_no_leakage`**: Enforces zero target leakage and validates harmonic trigonometric features fall strictly within [-1.0, 1.0].

To run the suite:
```bash
uv run pytest tests/test_pipeline.py -v
```

---

## 4. Real-Time Statistical Drift Detection

Production queries are logged asynchronously to `logs/predictions.jsonl`. AirIntel evaluates two complementary drift metrics:

### A. Population Stability Index (PSI)
Quantifies shifts in the continuous distribution of model inputs or predicted AQI between the baseline training distribution (<i>P</i>) and live serving queries (<i>Q</i>):

<div style="background: #F8FAFC; border: 1px solid #DBEAFE; border-left: 4px solid #1E3A8A; border-radius: 8px; padding: 14px 20px; margin: 14px 0; font-family: 'Cambria Math', 'Times New Roman', serif; text-align: center;">
  <div style="font-size: 18px; font-weight: 700; color: #1E3A8A; letter-spacing: 0.02em;">
    PSI = &sum;<sub><i>i</i>=1</sub><sup><i>K</i></sup> (<i>P<sub>i</sub></i> &minus; <i>Q<sub>i</sub></i>) &times; ln( <sup><i>P<sub>i</sub></i></sup>&frasl;<sub><i>Q<sub>i</sub></i></sub> )
  </div>
</div>

| PSI Threshold | Operational Status | System Directive |
| :--- | :--- | :--- |
| **PSI < 0.10** | **Green (Stable)** | Normal serving; no drift detected |
| **0.10 ≤ PSI < 0.25** | **Amber (Moderate Shift)** | Increase monitoring frequency; inspect microclimatic shifts |
| **PSI ≥ 0.25** | **Red (Significant Drift)** | Trigger automated model retraining alert |

### B. Two-Sample Kolmogorov-Smirnov (KS) Test
Tests whether the sample of live production queries <i>F</i>₁(<i>x</i>) originates from the same underlying distribution as the training baseline <i>F</i>₂(<i>x</i>):

<div style="background: #F8FAFC; border: 1px solid #DBEAFE; border-left: 4px solid #1E3A8A; border-radius: 8px; padding: 14px 20px; margin: 14px 0; font-family: 'Cambria Math', 'Times New Roman', serif; text-align: center;">
  <div style="font-size: 18px; font-weight: 700; color: #1E3A8A; letter-spacing: 0.02em;">
    <i>D</i> = sup<sub><i>x</i></sub> |<i>F</i>₁(<i>x</i>) &minus; <i>F</i>₂(<i>x</i>)|
  </div>
</div>

If the test statistic *p*-value falls below &alpha; = 0.05, a distribution shift alert is raised.

---

## 5. Docker Containerization & CI/CD

AirIntel is packaged as a multi-stage slim container (`Dockerfile`):
- Base image: `python:3.12-slim`
- User: Non-root user `appuser` (UID 1001) for strict runtime security
- Healthcheck probe: `curl -f http://localhost:8501/_stcore/health`
- Port exposure: Port 8501

Continuous Integration workflows (`.github/workflows/tests.yml` and `deploy.yml`) execute the automated test suite across Python versions on every commit and pull request.
