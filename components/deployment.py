import streamlit as st
import json
from pathlib import Path
from components.utils import load_table

def render_deployment(pipeline_bundle):
    """Render Stage 9: Deployment (Engineering proof, SLAs, and API contract specifications)."""
    st.markdown('<div class="editorial-title">Production Deployment Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-subtitle">High-throughput serialized scikit-learn pipeline bundle, measured latency SLAs, and OpenAPI REST contracts.</div>', unsafe_allow_html=True)
    
    # 1. SPECIFICATIONS & LATENCY SLAs
    st.markdown('<div class="editorial-section-heading">1. Pipeline Specifications & Measured Latency SLAs</div>', unsafe_allow_html=True)
    
    spec_c1, spec_c2, spec_c3, spec_c4 = st.columns(4)
    with spec_c1:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:16px; border-radius:8px;">
                <div style="font-size:12px; font-weight:700; color:#64748B; text-transform:uppercase;">Serialized Bundle</div>
                <div style="font-size:24px; font-weight:800; color:#0F172A; margin:4px 0;">3.57 MB</div>
                <div style="font-size:12.5px; color:#475569;">deployment_pipeline.pkl</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with spec_c2:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:16px; border-radius:8px;">
                <div style="font-size:12px; font-weight:700; color:#64748B; text-transform:uppercase;">Inference Latency</div>
                <div style="font-size:24px; font-weight:800; color:#10B981; margin:4px 0;">22 ms</div>
                <div style="font-size:12.5px; color:#475569;">P50 Median Latency</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with spec_c3:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:16px; border-radius:8px;">
                <div style="font-size:12px; font-weight:700; color:#64748B; text-transform:uppercase;">Tail SLA (P99)</div>
                <div style="font-size:24px; font-weight:800; color:#2563EB; margin:4px 0;">38 ms</div>
                <div style="font-size:12.5px; color:#475569;">P99 Latency SLA Target</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with spec_c4:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:16px; border-radius:8px;">
                <div style="font-size:12px; font-weight:700; color:#64748B; text-transform:uppercase;">Memory Footprint</div>
                <div style="font-size:24px; font-weight:800; color:#7C3AED; margin:4px 0;">~185 MB</div>
                <div style="font-size:12.5px; color:#475569;">Container Runtime RAM</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown('<div class="hairline-divider"></div>', unsafe_allow_html=True)
    
    # 2. DEPLOYMENT MODES AUDIT TABLE
    st.markdown('<div class="editorial-section-heading">2. Dual Inference Architecture Status</div>', unsafe_allow_html=True)
    dep_summary = load_table("deployment_summary.csv")
    if dep_summary is not None and not dep_summary.empty:
        st.dataframe(dep_summary, use_container_width=True, hide_index=True)
        
    st.markdown('<div class="hairline-divider"></div>', unsafe_allow_html=True)
    
    # 3. API SCHEMA CONTRACTS
    st.markdown('<div class="editorial-section-heading">3. OpenAPI & REST JSON Schema Contracts</div>', unsafe_allow_html=True)
    st.caption("Standardized contracts validated in `12_Deployment_Packaging.ipynb`:")
    
    req_c, res_c = st.columns(2)
    with req_c:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#1E293B; margin-bottom:6px;'>Sample API Request (POST /predict)</div>", unsafe_allow_html=True)
        sample_req = {
            "City": "Delhi",
            "Temp_2m_C": 35.0,
            "Humidity_Percent": 45.0,
            "Month": 11,
            "Hour": 18,
            "PM2_5_ugm3": 95.4,
            "PM10_ugm3": 178.2,
            "NO2_ugm3": 44.1,
            "SO2_ugm3": 18.2,
            "CO_ugm3": 850.0,
            "O3_ugm3": 54.0
        }
        st.code(json.dumps(sample_req, indent=2), language="json")
        
    with res_c:
        st.markdown("<div style='font-size:13px; font-weight:700; color:#1E293B; margin-bottom:6px;'>Sample API Response (Scientific Mode)</div>", unsafe_allow_html=True)
        sample_res = {
            "Prediction_Engine": "AirIntel v1.0",
            "Predicted_US_AQI": 157.44,
            "Predicted_AQI_Category": "Unhealthy",
            "Prediction_Confidence": 0.4994,
            "AirIntel_Risk_Score": 33.1,
            "Risk_Level": "Moderate",
            "Top_SHAP_Drivers": [
                {"Feature": "numerical__Northern_India", "SHAP_Impact": 25.03},
                {"Feature": "numerical__Temp_2m_C", "SHAP_Impact": 17.37},
                {"Feature": "categorical__Season_Monsoon", "SHAP_Impact": -13.91}
            ]
        }
        st.code(json.dumps(sample_res, indent=2), language="json")
        
    st.markdown('<div class="hairline-divider"></div>', unsafe_allow_html=True)
    
    # 4. DEPLOYMENT PIPELINE CODE
    st.markdown('<div class="editorial-section-heading">4. Python Integration Example</div>', unsafe_allow_html=True)
    st.caption("Direct python deserialization and scoring in under 5 lines:")
    py_code = """import pickle
import pandas as pd

# Load 3.57 MB serialized bundle
with open("models/deployment/deployment_pipeline.pkl", "rb") as f:
    bundle = pickle.load(f)

# Score payload
pipeline = bundle["pipeline"]
features = bundle["selected_features"]
df_input = pd.DataFrame([payload])[features]
predicted_aqi = pipeline.predict(df_input)[0]
"""
    st.code(py_code, language="python")
