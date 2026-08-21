import streamlit as st
import pickle
import json
from pathlib import Path
from datetime import datetime, timezone

@st.cache_resource
def load_deployment_bundle(bundle_path="models/deployment/deployment_pipeline.pkl"):
    """Safely load and cache the deployment pipeline bundle."""
    path = Path(bundle_path)
    if not path.exists():
        return None, f"Deployment bundle file not found at '{bundle_path}'. Ensure Notebook 12 exports have executed."
    
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return bundle, None
    except Exception as e:
        return None, f"Failed to deserialize deployment bundle: {str(e)}"

def format_iso_timestamp():
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def get_system_health_status(pipeline_bundle):
    """Evaluate operational readiness of pipeline components."""
    if not pipeline_bundle:
        return {
            "Pipeline_Bundle": "Failed",
            "Regression_Model": "Unavailable",
            "Classifier_Model": "Unavailable",
            "Metadata": "Unavailable",
            "Engine_State": "Offline"
        }
    
    reg_ok = "reg_pipeline" in pipeline_bundle and pipeline_bundle["reg_pipeline"] is not None
    cls_ok = "cls_pipeline" in pipeline_bundle and pipeline_bundle["cls_pipeline"] is not None
    meta_ok = "selected_features" in pipeline_bundle and "valid_cities" in pipeline_bundle
    
    return {
        "Pipeline_Bundle": "Loaded" if pipeline_bundle else "Failed",
        "Regression_Model": "Loaded" if reg_ok else "Failed",
        "Classifier_Model": "Loaded" if cls_ok else "Failed",
        "Metadata": "Loaded" if meta_ok else "Failed",
        "Engine_State": "Ready" if (reg_ok and cls_ok and meta_ok) else "Degraded"
    }
