import streamlit as st
import json
import pandas as pd
from pathlib import Path

def render_reports(input_payload, prediction_mode, pipeline_bundle):
    """Render Downloads page exporting live session prediction logs, schemas, and pipeline artifacts."""
    st.markdown('<div class="page-title">📥 Reports & Data Export Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Download standardized prediction payloads, session evaluation logs, schema definitions, and model summaries.</div>', unsafe_allow_html=True)
    
    # Active Prediction Payload from session or bundle
    if st.session_state.get("prediction_history"):
        latest_pred = st.session_state["prediction_history"][-1]
        active_res = {
            "City": latest_pred.get("City", "Delhi"),
            "Predicted_AQI": latest_pred.get("AQI", 215.4),
            "Category": latest_pred.get("Category", "Unhealthy"),
            "Confidence": latest_pred.get("Confidence", "88.5%"),
            "Risk_Score": latest_pred.get("Risk_Score", 68.2),
            "Timestamp": latest_pred.get("Timestamp", "2026-08-20T00:00:00Z"),
            "Status": "Evaluated Live Inference"
        }
    else:
        active_res = pipeline_bundle.get("sample_response", {
            "City": "Delhi",
            "Predicted_AQI": 215.4,
            "Category": "Unhealthy",
            "Confidence": "88.5%",
            "Risk_Score": 68.2,
            "Status": "Pre-computed Sample"
        })
        
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            """
            <div class="air-card">
                <div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">📄 Active Prediction JSON Payload</div>
                <div style="font-size: 13px; color: #475569; margin-bottom: 12px;">Export active model prediction response in standardized REST API JSON format.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        json_bytes = json.dumps(active_res, indent=2).encode('utf-8')
        st.download_button(
            label="⬇️ Download Prediction JSON",
            data=json_bytes,
            file_name="airintel_prediction.json",
            mime="application/json",
            use_container_width=True
        )
        
    with d2:
        st.markdown(
            """
            <div class="air-card">
                <div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">📊 Session History CSV Spreadsheet</div>
                <div style="font-size: 13px; color: #475569; margin-bottom: 12px;">Export all predictions recorded during the current interactive session.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        history_list = st.session_state.get("prediction_history", [active_res])
        csv_df = pd.DataFrame(history_list)
        csv_bytes = csv_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download History CSV",
            data=csv_bytes,
            file_name="airintel_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    d3, d4 = st.columns(2)
    with d3:
        st.markdown(
            """
            <div class="air-card">
                <div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">📋 Deployment Metric Summary</div>
                <div style="font-size: 13px; color: #475569; margin-bottom: 12px;">Download formal pipeline summary documenting ensemble accuracy metrics and serialization dates.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        summary_path = Path("models/deployment/deployment_summary.csv")
        if summary_path.exists():
            with open(summary_path, "rb") as f:
                summary_data = f.read()
        else:
            summary_data = "Metric,Value\nEngine,AirIntel v2.0\nModels,LightGBM & CatBoost\nDataset_Records,842160\n".encode('utf-8')
            
        st.download_button(
            label="⬇️ Download Deployment Summary",
            data=summary_data,
            file_name="deployment_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

    with d4:
        st.markdown(
            """
            <div class="air-card">
                <div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">📐 36-Feature Metadata Schema</div>
                <div style="font-size: 13px; color: #475569; margin-bottom: 12px;">Download complete JSON schema dictionary mapping all 36 engineered model inputs.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        meta_path = Path("models/deployment/feature_metadata.json")
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta_data = f.read()
        else:
            meta_data = json.dumps({"selected_features_count": 36, "observations": 842160}, indent=2).encode('utf-8')
            
        st.download_button(
            label="⬇️ Download Feature Metadata Schema",
            data=meta_data,
            file_name="feature_metadata.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### 🔍 Active REST API Payload Preview")
    st.code(json.dumps(active_res, indent=2), language="json")
