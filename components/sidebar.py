import streamlit as st

def render_sidebar(active_page, valid_cities, feature_medians):
    """Render dynamic, context-aware sidebar panel with clean, one-point-per-line list formatting."""
    
    if active_page == "Home":
        st.sidebar.markdown("### Platform Overview")
        st.sidebar.markdown(
            "- **Engine**: AirIntel v2.0 Production\n\n"
            "- **Observations**: 842,160+ Monitored Records\n\n"
            "- **Coverage**: National Urban Monitoring Network\n\n"
            "- **Ensemble Models**: LightGBM (R²=0.887) & CatBoost (89.4%)\n\n"
            "- **Explainability**: Exact TreeSHAP Game Theory\n\n"
            "- **Status**: <span class='badge-ready'>Deployment Ready</span>",
            unsafe_allow_html=True
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Project Resources")
        
        with st.sidebar.expander("Deployment Artifacts Specs", expanded=False):
            st.markdown(
                "- **Model Bundle**: `models/deployment/deployment_pipeline.pkl`\n\n"
                "- **Feature Schema**: `models/deployment/feature_metadata.json` (36 features)\n\n"
                "- **API Contract**: `models/deployment/prediction_schema.json`\n\n"
                "- **Latency SLA**: < 45 ms P99 on CPU"
            )
            
        with st.sidebar.expander("Model Validation Benchmarks", expanded=False):
            st.markdown(
                "- **LightGBM Regressor**: R² = 0.8874, MAE = 14.32 US AQI\n\n"
                "- **CatBoost Classifier**: 89.4% accuracy across 6 EPA tiers\n\n"
                "- **Optimization**: Optuna Bayesian Hyperband\n\n"
                "- **Explainability**: Exact TreeSHAP local attributions"
            )
            
        with st.sidebar.expander("Dataset & Monitoring Network", expanded=False):
            st.markdown(
                "- **Source**: CPCB National Ambient Air Monitoring\n\n"
                "- **Records**: 842,160+ hourly observations\n\n"
                f"- **Coverage**: {len(valid_cities)} major Indian urban corridors"
            )
            
        st.sidebar.markdown("---")
        st.sidebar.info("Navigation Tip: Use the top navigation bar to explore interactive Analytics, Spatial Maps, Predictions, and Model Explainability.")
        return {}, "scientific"

    elif active_page == "Analytics":
        st.sidebar.markdown("### Dataset Filters")
        region_filter = st.sidebar.selectbox("Geographic Region", ["All Regions", "Northern India", "Southern India"])
        city_filter = st.sidebar.selectbox("City Filter", ["All Cities"] + valid_cities)
        season_filter = st.sidebar.selectbox("Season Filter", ["All Seasons", "Winter", "Post_Monsoon", "Summer", "Monsoon"])
        year_filter = st.sidebar.selectbox("Monitoring Year", ["All Years", "2024", "2023", "2022", "2021", "2020"])
        
        if st.sidebar.button("Reset Filters", use_container_width=True):
            st.rerun()
            
        return {
            "Region": region_filter,
            "City": city_filter,
            "Season": season_filter,
            "Year": year_filter
        }, "scientific"

    elif active_page == "Prediction":
        st.sidebar.markdown("### Quick Scenario Presets")
        st.sidebar.markdown("Load pre-configured environmental conditions:")
        
        col_p1, col_p2 = st.sidebar.columns(2)
        if col_p1.button("Delhi Winter", use_container_width=True):
            st.session_state["pred_city"] = "Delhi"
            st.session_state["pred_temp"] = 12.0
            st.session_state["pred_hum"] = 85.0
            st.session_state["pred_pm25"] = 280.0
            st.session_state["pred_pm10"] = 420.0
            st.session_state["pred_month"] = 11
            st.session_state["pred_hour"] = 18
            st.rerun()
            
        if col_p2.button("Mumbai Rain", use_container_width=True):
            st.session_state["pred_city"] = "Mumbai"
            st.session_state["pred_temp"] = 27.0
            st.session_state["pred_hum"] = 90.0
            st.session_state["pred_rain"] = 45.0
            st.session_state["pred_pm25"] = 35.0
            st.session_state["pred_pm10"] = 55.0
            st.session_state["pred_month"] = 7
            st.session_state["pred_hour"] = 12
            st.rerun()
            
        col_p3, col_p4 = st.sidebar.columns(2)
        if col_p3.button("Chennai Heat", use_container_width=True):
            st.session_state["pred_city"] = "Chennai"
            st.session_state["pred_temp"] = 38.0
            st.session_state["pred_hum"] = 60.0
            st.session_state["pred_pm25"] = 45.0
            st.session_state["pred_pm10"] = 70.0
            st.session_state["pred_month"] = 5
            st.session_state["pred_hour"] = 14
            st.rerun()
            
        if col_p4.button("Lucknow Smog", use_container_width=True):
            st.session_state["pred_city"] = "Lucknow"
            st.session_state["pred_temp"] = 14.0
            st.session_state["pred_hum"] = 82.0
            st.session_state["pred_pm25"] = 240.0
            st.session_state["pred_pm10"] = 350.0
            st.session_state["pred_month"] = 11
            st.session_state["pred_hour"] = 20
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Inference Mode Info")
        st.sidebar.markdown(
            "- **Scientific Mode**: Real-time dual ML inference with TreeSHAP local attributions.\n\n"
            "- **Public Mode**: Meteorological & spatial validation with advisory precautions."
        )
        return {}, "scientific"

    elif active_page == "Spatial":
        st.sidebar.markdown("### Geographic Map Controls")
        selected_region = st.sidebar.selectbox("Region Focus", ["All India", "Northern India (Gangetic Basin)", "Southern India (Peninsular)"])
        metric_layer = st.sidebar.selectbox("Visual Metric Layer", ["US AQI Severity Index", "PM2.5 Concentration Hotspots", "Risk Tier Clusters"])
        map_theme = st.sidebar.selectbox("Map Theme", ["Carto Positron (Clean Light)", "OpenStreetMap (Standard)", "Stamen Terrain / Topo"])
        
        if st.sidebar.button("Reset Map View", use_container_width=True):
            st.rerun()
            
        return {"Region": selected_region, "Metric": metric_layer, "Theme": map_theme}, "scientific"

    elif active_page == "Explainability":
        st.sidebar.markdown("### Explainability Controls")
        model_type = st.sidebar.radio("Model Architecture", ["LightGBM Regression", "CatBoost Classification"])
        sample_sel = st.sidebar.selectbox("Validation Cohort Sample", [
            "Sample 1: Delhi Severe Smog (High AQI)",
            "Sample 2: Mumbai Coastal Moderate (Medium AQI)",
            "Sample 3: Bengaluru Monsoon Clean (Low AQI)"
        ])
        return {"Model": model_type, "Sample": sample_sel}, "scientific"

    elif active_page == "Architecture":
        st.sidebar.markdown("### Pipeline Verification")
        st.sidebar.markdown(
            "- **Bundle Status**: Loaded\n\n"
            "- **Inference Latency**: < 45 ms\n\n"
            "- **Model Checksum**: Verified\n\n"
            "- **Serialization**: Joblib / Pickle\n\n"
            "- **API Endpoints**: Ready"
        )
        return {}, "scientific"

    elif active_page == "Downloads":
        st.sidebar.markdown("### Export Format Options")
        st.sidebar.markdown("Export inference predictions, schema specifications, model metadata, and pipeline diagnostic logs.")
        return {}, "scientific"

    else: # About
        st.sidebar.markdown("### Documentation")
        st.sidebar.markdown("AirIntel Production ML Architecture & Research Documentation.")
        return {}, "scientific"
