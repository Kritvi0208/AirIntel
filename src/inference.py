"""
AirIntel - Production Real-Time Dual Inference Serving Module
Author: Ritvika (IIT BHU)
Provides low-latency (<50ms) inference with dynamic imputation and structured JSON logging.
"""

import os
import time
import json
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "deployment", "deployment_pipeline.pkl")
LOGS_PATH = os.path.join(BASE_DIR, "logs", "predictions.jsonl")

# Cached pipeline instance
_CACHED_BUNDLE = None

def get_pipeline():
    """Singleton getter for deployment pipeline bundle to eliminate reload latency."""
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model bundle not found at {MODEL_PATH}")
        _CACHED_BUNDLE = joblib.load(MODEL_PATH)
    return _CACHED_BUNDLE

def format_tier_name(tier_code):
    """Normalize internal tier labels to official EPA display names."""
    mapping = {
        "Good": "Good",
        "Moderate": "Moderate",
        "Unhealthy_Sensitive": "Unhealthy for Sensitive Groups",
        "Unhealthy": "Unhealthy",
        "Very_Unhealthy": "Very Unhealthy",
        "Hazardous": "Hazardous"
    }
    return mapping.get(tier_code, str(tier_code))

def predict(input_payload: dict, mode: str = "scientific", log_prediction: bool = True) -> dict:
    """
    Execute synchronized dual inference (LightGBM AQI + CatBoost Tier).
    
    Parameters:
    -----------
    input_payload : dict
        Raw sensor / citizen input values.
    mode : str
        'scientific' (full criteria pollutants) or 'public' (weather-driven).
    log_prediction : bool
        Whether to append inference telemetry to logs/predictions.jsonl.
        
    Returns:
    --------
    dict
        Structured prediction result with latency metrics.
    """
    t_start = time.perf_counter()
    bundle = get_pipeline()
    
    reg_pipe = bundle['reg_pipeline']
    cls_pipe = bundle['cls_pipeline']
    label_encoder = bundle['label_encoder']
    selected_features = bundle['selected_features']
    medians = bundle['feature_medians']
    city_coords = bundle['city_coords']
    valid_cities = bundle['valid_cities']
    
    city = input_payload.get("City", valid_cities[0] if valid_cities else "Delhi")
    month = int(input_payload.get("Month", 11))
    hour = int(input_payload.get("Hour", 18))
    
    # 1. Resolve Geographic & Cyclical features
    coords = city_coords.get(city, {"Latitude": medians.get("Latitude", 28.61), "Longitude": medians.get("Longitude", 77.20)})
    lat = float(coords.get("Latitude", 28.61))
    lon = float(coords.get("Longitude", 77.20))
    
    d = dict(input_payload)
    d['City'] = city
    d['Latitude'] = lat
    d['Longitude'] = lon
    d['Absolute_Latitude'] = abs(lat)
    d['Lat_Long_Interaction'] = lat * lon
    d['Northern_India'] = 1 if lat > 20.0 else 0
    d['Month_Sin'] = math.sin(2 * math.pi * month / 12)
    d['Month_Cos'] = math.cos(2 * math.pi * month / 12)
    d['Hour_Cos'] = math.cos(2 * math.pi * hour / 24)
    d['Weekday_Sin'] = float(input_payload.get('Weekday_Sin', 0.5))
    d['Weekday_Cos'] = float(input_payload.get('Weekday_Cos', 0.86))
    
    temp = float(input_payload.get("Temp_2m_C", medians.get("Temp_2m_C", 25.0)))
    hum = float(input_payload.get("Humidity_Percent", medians.get("Humidity_Percent", 55.0)))
    d['Temp_2m_C'] = temp
    d['Humidity_Percent'] = hum
    d['Temp_Humidity'] = temp * hum
    
    # Impute missing feature columns using training consensus medians
    for col in selected_features:
        if col not in d or d[col] is None or (isinstance(d[col], float) and np.isnan(d[col])):
            d[col] = medians.get(col, 0.0)
            
    # Form 1-row DataFrame aligned to feature order
    df_in = pd.DataFrame([{col: d[col] for col in selected_features}])
    
    # 2. Execute Dual Inference
    pred_aqi_raw = reg_pipe.predict(df_in)[0]
    pred_aqi = max(0.0, min(500.0, float(pred_aqi_raw)))
    
    pred_tier_idx = cls_pipe.predict(df_in)[0]
    if isinstance(pred_tier_idx, (list, np.ndarray)):
        pred_tier_idx = pred_tier_idx[0]
    tier_raw_name = label_encoder.inverse_transform([int(pred_tier_idx)])[0]
    tier_name = format_tier_name(tier_raw_name)
    
    # Class probability distribution
    probs = cls_pipe.predict_proba(df_in)[0]
    class_probs = {format_tier_name(cls): round(float(p), 4) for cls, p in zip(label_encoder.classes_, probs)}
    
    latency_ms = (time.perf_counter() - t_start) * 1000.0
    
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city": city,
        "mode": mode,
        "predicted_aqi": round(pred_aqi, 1),
        "predicted_tier": tier_name,
        "tier_probabilities": class_probs,
        "latency_ms": round(latency_ms, 2)
    }
    
    # 3. Telemetry Logging to predictions.jsonl
    if log_prediction:
        try:
            os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
            with open(LOGS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
        except Exception:
            pass  # Non-blocking logging failure
            
    return result

if __name__ == "__main__":
    test_input = {
        "City": "Delhi",
        "Month": 11,
        "Hour": 18,
        "Temp_2m_C": 26.5,
        "Humidity_Percent": 62.0,
        "Surface_Pressure_hPa": 1012.0,
        "Wind_Speed_10m_kmh": 6.5,
        "Rain_mm": 0.0,
        "PM2.5": 142.0,
        "PM10": 210.0,
        "NO2": 45.0,
        "SO2": 18.0,
        "CO": 1.4,
        "O3": 38.0
    }
    out = predict(test_input, mode="scientific")
    print("AirIntel Production Serving Output:")
    print(json.dumps(out, indent=2))
