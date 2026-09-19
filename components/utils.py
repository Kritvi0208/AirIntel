import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

@st.cache_resource
def load_deployment_bundle(bundle_path="models/deployment/deployment_pipeline.pkl"):
    """Load and cache the trained serialized production deployment bundle."""
    path = Path(bundle_path)
    if not path.exists():
        return None, f"Deployment bundle missing at {bundle_path}."
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle, None
    except Exception as e:
        return None, f"Error deserializing deployment bundle: {str(e)}"

@st.cache_data
def load_table(filename):
    """Programmatically load and cache a report CSV table with verification."""
    p = Path(f"reports/tables/{filename}")
    if p.exists():
        try:
            return pd.read_csv(p, encoding="utf-8")
        except Exception:
            try:
                return pd.read_csv(p, encoding="latin1")
            except Exception:
                return None
    return None

@st.cache_data
def load_clean_dataset(sample_size=40000):
    """Load processed clean Parquet dataset with standardized canonical column mapping."""
    p = Path("data/processed/clean_airintel.parquet")
    if p.exists():
        try:
            df = pd.read_parquet(p)
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            # Map canonical columns
            col_map = {
                "US_AQI": "AQI",
                "PM2_5_ugm3": "PM2.5",
                "PM10_ugm3": "PM10",
                "NO2_ugm3": "NO2",
                "SO2_ugm3": "SO2",
                "CO_ugm3": "CO",
                "O3_ugm3": "O3",
                "Temp_2m_C": "Temp",
                "Humidity_Percent": "Humidity",
                "Wind_Speed_10m_kmh": "Wind",
                "Rain_mm": "Rain",
                "Surface_Pressure_hPa": "Pressure"
            }
            for orig, clean in col_map.items():
                if orig in df.columns and clean not in df.columns:
                    df[clean] = df[orig]
            return df
        except Exception:
            return None
    return None

def get_verified_metric(table_name, filter_col, filter_val, target_col, fallback=None):
    """Programmatically query an exact metric value from a verified table artifact."""
    df = load_table(table_name)
    if df is not None and filter_col in df.columns and target_col in df.columns:
        row = df[df[filter_col].astype(str).str.contains(str(filter_val), case=False, na=False)]
        if not row.empty:
            return row.iloc[0][target_col]
    return fallback
