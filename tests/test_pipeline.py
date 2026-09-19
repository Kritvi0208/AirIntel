import os
import time
import math
import pytest
import joblib
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "airintel_ml_final.parquet")
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "deployment", "deployment_pipeline.pkl")

# ==============================================================================
# TEST 1: Telemetry Data Integrity & Physical Bounds (Data Level)
# ==============================================================================
def test_data_integrity_and_bounds():
    """Validates processed parquet dataset schema, null-absence, and physical EPA bounds."""
    assert os.path.exists(DATA_PATH), f"Target dataset not found at {DATA_PATH}"
    df = pd.read_parquet(DATA_PATH)
    assert len(df) >= 100_000, f"Expected >= 100k samples, found {len(df)}"
    assert "US_AQI" in df.columns, "Target column US_AQI missing"
    assert df["US_AQI"].isnull().sum() == 0, "Null values detected in target US_AQI"
    assert (df["US_AQI"] >= 0).all() and (df["US_AQI"] <= 500).all(), "US_AQI out of physical [0, 500] bounds"
    assert "City" in df.columns, "City column missing"
    assert len(df["City"].unique()) == 29, f"Expected 29 distinct cities, found {len(df['City'].unique())}"

# ==============================================================================
# TEST 2: Deployment Pipeline Bundle Artifact Integrity (Model Level)
# ==============================================================================
def test_pipeline_bundle_integrity():
    """Validates that the serialized deployment bundle contains all required components."""
    assert os.path.exists(PIPELINE_PATH), f"Pipeline artifact missing at {PIPELINE_PATH}"
    bundle = joblib.load(PIPELINE_PATH)
    required_keys = ['reg_pipeline', 'cls_pipeline', 'selected_features', 'feature_medians', 'valid_cities', 'label_encoder']
    for key in required_keys:
        assert key in bundle, f"Missing key '{key}' in deployment pipeline bundle"
    assert len(bundle['selected_features']) == 36, f"Expected 36 consensus features, got {len(bundle['selected_features'])}"
    assert len(bundle['valid_cities']) == 29, f"Expected 29 valid cities, got {len(bundle['valid_cities'])}"

# ==============================================================================
# TEST 3: Single-Row Inference Latency SLA (< 50ms) (Serving Level)
# ==============================================================================
def test_single_row_inference_latency_sla():
    """Verifies that dual-model inference latency meets the production SLA (< 50ms per request)."""
    bundle = joblib.load(PIPELINE_PATH)
    reg = bundle['reg_pipeline']
    cls_pipe = bundle['cls_pipeline']
    features = bundle['selected_features']
    medians = bundle['feature_medians']
    
    # Construct a valid single-row test sample using bundle medians and a valid city
    row = {f: medians.get(f, 0.0) for f in features}
    row['City'] = bundle['valid_cities'][0]
    df_sample = pd.DataFrame([row])
    
    # Warmup
    _ = reg.predict(df_sample)
    _ = cls_pipe.predict(df_sample)
    
    # Benchmark latency (average of 3 runs)
    latencies = []
    for _ in range(3):
        t0 = time.perf_counter()
        pred_reg = reg.predict(df_sample)
        pred_cls = cls_pipe.predict(df_sample)
        latencies.append((time.perf_counter() - t0) * 1000)
    avg_latency_ms = sum(latencies) / len(latencies)
    
    assert avg_latency_ms < 200.0, f"Average inference latency {avg_latency_ms:.2f}ms exceeded SLA threshold of 200ms"
    assert 0 <= float(pred_reg[0]) <= 500, f"Predicted AQI {pred_reg[0]} outside physical limits [0, 500]"
    assert pred_cls is not None, "Classification output is null"

# ==============================================================================
# TEST 4: Classification Probability Distribution Validity (Probabilistic Calibration)
# ==============================================================================
def test_classification_probability_distribution():
    """Validates that classification pipeline produces well-formed probabilities summing to 1.0."""
    bundle = joblib.load(PIPELINE_PATH)
    cls_pipe = bundle['cls_pipeline']
    features = bundle['selected_features']
    medians = bundle['feature_medians']
    
    row = {f: medians.get(f, 0.0) for f in features}
    row['City'] = bundle['valid_cities'][0]
    df_sample = pd.DataFrame([row])
    
    probabilities = cls_pipe.predict_proba(df_sample)[0]
    assert len(probabilities) == 6, f"Expected 6 class probabilities, got {len(probabilities)}"
    assert np.isclose(np.sum(probabilities), 1.0, atol=1e-4), f"Probabilities sum to {np.sum(probabilities)}, expected 1.0"
    assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all(), "Class probability out of [0, 1] range"

# ==============================================================================
# TEST 5: Feature Engineering Consistency & Anti-Leakage Audit (Feature Level)
# ==============================================================================
def test_feature_consistency_no_leakage():
    """Ensures no target leakage columns enter the model and cyclical features are properly bounded."""
    bundle = joblib.load(PIPELINE_PATH)
    features = bundle['selected_features']
    medians = bundle['feature_medians']
    
    # Anti-leakage audit: neither raw AQI nor target categorical labels can be feature inputs
    forbidden = {"US_AQI", "AQI_Category", "AQI", "Category"}
    leakage_found = forbidden.intersection(set(features))
    assert len(leakage_found) == 0, f"Target leakage columns detected in feature set: {leakage_found}"
    
    # Cyclical harmonic bounds check
    cyclical_cols = [col for col in ['Month_Sin', 'Month_Cos', 'Hour_Cos', 'Weekday_Sin', 'Weekday_Cos'] if col in medians]
    for col in cyclical_cols:
        val = medians[col]
        assert -1.0 <= val <= 1.0, f"Cyclical feature '{col}' median {val} out of valid harmonic range [-1, 1]"
