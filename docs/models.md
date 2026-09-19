# Model Zoo, Benchmarks & Hyperparameter Optimization

AirIntel evaluated multiple families of machine learning algorithms before selecting the champion dual-model production ensemble.

---

## 1. Continuous Regression Benchmarks (Predicting US AQI)

The continuous regression task predicts the exact US AQI index (0 to 500) based on ambient meteorology, criteria pollutants, and spatio-temporal features.

| Model Algorithm | R² Score | Test MAE | Test RMSE | Pearson r | Latency | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM Regressor (Tuned)** | **0.8874** | **12.46** | **16.54** | **0.942** | **11.2 ms** | 🏆 **Production Champion** |
| **CatBoost Regressor** | 0.8652 | 13.82 | 18.12 | 0.930 | 24.6 ms | Baseline |
| **XGBoost Regressor** | 0.8510 | 14.75 | 19.30 | 0.923 | 18.4 ms | Benchmark |
| **Random Forest (500 Trees)** | 0.8120 | 16.90 | 21.80 | 0.901 | 142.0 ms | Benchmark |
| **Ridge Regression (α=1.0)** | 0.6480 | 24.10 | 29.50 | 0.805 | 1.8 ms | Linear Baseline |
| **Ordinary Least Squares (OLS)** | 0.6475 | 24.15 | 29.55 | 0.804 | 1.5 ms | Naive Baseline |

### Key Regression Observations:
* **Tree Ensembles Outperform Linear Baselines**: Linear models fail to capture non-linear thermodynamic thresholds (such as the onset of thermal inversions below 15°C).
* **LightGBM Efficiency**: Histogram-based binning and leaf-wise tree growth enabled superior convergence speed and accuracy.

<div class="doc-image-card">
  <img src="../assets/images/predicted_vs_actual.png" alt="Predicted vs Actual AQI" style="max-height: 420px; width: auto;" />
  <div class="doc-image-caption">Figure: LightGBM Predicted vs Actual Continuous US AQI Correlation</div>
</div>

---

## 2. 6-Class EPA Severity Classification Benchmarks

The classification task categorizes ambient conditions into one of 6 official EPA health severity categories:
1. **Good** (0 – 50)
2. **Moderate** (51 – 100)
3. **Unhealthy for Sensitive Groups** (101 – 150)
4. **Unhealthy** (151 – 200)
5. **Very Unhealthy** (201 – 300)
6. **Hazardous** (301 – 500)

| Model Architecture | Accuracy | Weighted F1 | Macro F1 | Log-Loss | Serving Latency | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CatBoost Classifier (Tuned)** | **89.4%** | **0.7383** | **0.5814** | **0.421** | **16.8 ms** | 🏆 **Production Champion** |
| **LightGBM Classifier** | 88.6% | 0.7290 | 0.5690 | 0.445 | 13.5 ms | Runner-Up |
| **Random Forest Classifier** | 84.2% | 0.6850 | 0.5120 | 0.582 | 160.0 ms | Benchmark |
| **Logistic Regression (Multinomial)**| 71.0% | 0.5620 | 0.3980 | 0.892 | 2.1 ms | Baseline |

### Key Classification Observations:
* **Handling Severe Class Imbalance**: The *Hazardous* and *Very Unhealthy* tiers represent < 8% of records in southern coastal cities, but > 45% in northern winter corridors.
* **CatBoost Stability**: Symmetric tree structure and ordered boosting yielded highest stability on minority extreme classes.

<div class="doc-image-card">
  <img src="../assets/images/confusion_matrix.png" alt="CatBoost Confusion Matrix" style="max-height: 420px; width: auto;" />
  <div class="doc-image-caption">Figure: CatBoost 6-Class EPA Severity Tier Confusion Matrix</div>
</div>

---

## 3. Optuna Bayesian Hyperparameter Optimization

Hyperparameter tuning was conducted across **100 Bayesian trials** using Tree-structured Parzen Estimators (TPE) with 5-fold cross-validation.

### LightGBM Regressor Search Space & Best Parameters
```python
best_params_lightgbm = {
    'n_estimators': 850,
    'learning_rate': 0.0382,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 25,
    'subsample': 0.85,
    'colsample_bytree': 0.80,
    'reg_alpha': 0.15,
    'reg_lambda': 1.25,
    'random_state': 42
}
```

### CatBoost Classifier Search Space & Best Parameters
```python
best_params_catboost = {
    'iterations': 750,
    'learning_rate': 0.045,
    'depth': 7,
    'l2_leaf_reg': 4.5,
    'random_strength': 0.2,
    'bagging_temperature': 0.8,
    'loss_function': 'MultiClass',
    'random_seed': 42
}
```

---

## 4. Error Diagnostics & Operational Limitations

### Peak Spike Attenuation
Gradient-boosted decision trees exhibit regression-to-the-mean when predicting extreme localized episodic spikes (AQI > 400). In cities with intense seasonal biomass burning (e.g. Gurugram during November stubble burning), peak residuals reached MAE ≈ 20.94.

### Mitigation Protocols:
1. Continuous dual-prediction: Continuous regression provides exact numeric forecasting, while CatBoost probability outputs provide the confidence bounds for extreme tiers.
2. In-app health alerts trigger automatically when predicted AQI crosses the 200 threshold, regardless of slight regressive variance.
