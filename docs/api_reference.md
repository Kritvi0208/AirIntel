# Python API & Serving Reference

This document outlines the core internal Python APIs for inference serving, drift detection, and telemetry tracking.

---

## 1. Serving & Inference Engine (`src/inference.py`)

The primary serving interface encapsulates feature alignment, coordinate mapping, and dual-model prediction.

### Function: `predict()`
```python
def predict(
    input_payload: dict,
    mode: str = "scientific",
    log_prediction: bool = True
) -> dict:
```

#### Parameters:
- `input_payload` (*dict*): Dictionary of meteorological and pollutant measurements.
  - Required keys (Scientific Mode): `City`, `Month`, `Hour`, `Temp_2m_C`, `Humidity_Percent`, `Surface_Pressure_hPa`, `Wind_Speed_10m_kmh`, `Rain_mm`, `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`.
  - Required keys (Public Citizen Mode): `City`, `Month`, `Hour`, `Temp_2m_C`, `Humidity_Percent`, `Wind_Speed_10m_kmh`, `Rain_mm`. Unmeasured criteria pollutants automatically fall back to station historical medians.
- `mode` (*str*, optional): Serving architecture mode. Either `"scientific"` or `"public"`. Defaults to `"scientific"`.
- `log_prediction` (*bool*, optional): Whether to record the query and prediction asynchronously into `logs/predictions.jsonl`. Defaults to `True`.

#### Returns:
- `dict`:
  - `predicted_aqi` (*float*): Continuous predicted US AQI value (0.0 to 500.0).
  - `predicted_severity` (*str*): EPA severity tier string (e.g. `"Moderate"`, `"Unhealthy"`).
  - `confidence_score` (*float*): CatBoost softmax probability for the winning class (0.0 to 1.0).
  - `inference_time_ms` (*float*): Single-row execution time in milliseconds.

#### Example Usage:
```python
from src.inference import predict

payload = {
    "City": "Delhi",
    "Month": 12,
    "Hour": 21,
    "Temp_2m_C": 11.5,
    "Humidity_Percent": 84.0,
    "Surface_Pressure_hPa": 1018.0,
    "Wind_Speed_10m_kmh": 4.2,
    "Rain_mm": 0.0,
    "PM2.5": 285.0,
    "PM10": 420.0,
    "NO2": 68.0,
    "SO2": 24.0,
    "CO": 2.8,
    "O3": 32.0
}

response = predict(payload, mode="scientific")
print(f"Predicted AQI: {response['predicted_aqi']:.1f}")
print(f"EPA Tier: {response['predicted_severity']}")
print(f"Confidence: {response['confidence_score']*100:.1f}%")
```

---

## 2. Statistical Drift Monitoring (`src/monitoring.py`)

Provides analytical routines for detecting feature and prediction drift.

### Function: `calculate_psi()`
```python
def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10
) -> float:
```
Computes the Population Stability Index between reference training array and production sample.

### Function: `run_drift_audit()`
```python
def run_drift_audit(
    telemetry_log_path: str = "logs/predictions.jsonl",
    baseline_data_path: str = "data/processed/airintel_ml_final.parquet"
) -> dict:
```
Performs an automated audit comparing live telemetry logs against baseline parquet data, returning PSI, KS test statistics, and drift alerts.

---

## 3. Pipeline Bundle Loader (`components/utils.py`)

### Function: `load_deployment_bundle()`
```python
@st.cache_resource
def load_deployment_bundle(filepath: str) -> tuple[dict, str | None]:
```
Safely unpickles the consolidated deployment bundle `models/deployment/deployment_pipeline.pkl` into Streamlit memory cache.

#### Dictionary Keys in Bundle:
- `reg_pipeline`: Fitted LightGBM scikit-learn pipeline.
- `cls_pipeline`: Fitted CatBoost scikit-learn pipeline.
- `label_encoder`: Fitted `LabelEncoder` mapping integer classes to EPA severity strings.
- `selected_features`: List of 36 consensus feature names in exact column order.
- `valid_cities`: List of 29 supported Indian urban monitoring centers.
- `city_coords`: Mapping of city names to `{"Latitude": float, "Longitude": float}`.
- `feature_medians`: Dictionary of nationwide median values for robust fallback imputation.
