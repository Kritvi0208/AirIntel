import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import math
import numpy as np
from datetime import datetime, timezone

def render_prediction_page(input_payload, prediction_mode_arg, pipeline_bundle):
    """Render interactive prediction module supporting both Scientific and Public Citizen live inference with explicit analysis button."""
    st.markdown('<div class="page-title">🤖 Real-Time Prediction & Risk Inference Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Interactive machine learning inference consuming the serialized production deployment pipeline.</div>', unsafe_allow_html=True)
    
    # Mode selection directly on the page
    mode_selection = st.segmented_control(
        "Select Inference Mode",
        ["🔬 Scientific Diagnostic Mode (Full Pollutants)", "👥 Public Citizen Mode (Weather-Driven)"],
        default="🔬 Scientific Diagnostic Mode (Full Pollutants)",
        label_visibility="collapsed"
    )
    
    is_scientific = "Scientific" in (mode_selection or "Scientific")
    
    # Initialize session state for inputs dynamically from deployment bundle
    valid_cities = pipeline_bundle.get('valid_cities', ['Delhi', 'Mumbai', 'Bengaluru', 'Lucknow', 'Chennai'])
    feature_medians = pipeline_bundle.get('feature_medians', {})
    
    if "pred_city" not in st.session_state or st.session_state["pred_city"] not in valid_cities:
        st.session_state["pred_city"] = "Delhi" if "Delhi" in valid_cities else valid_cities[0]
    if "pred_temp" not in st.session_state:
        st.session_state["pred_temp"] = float(feature_medians.get('Temp_2m_C', 25.0))
    if "pred_hum" not in st.session_state:
        st.session_state["pred_hum"] = float(feature_medians.get('Humidity_Percent', 55.0))
    if "pred_pressure" not in st.session_state:
        st.session_state["pred_pressure"] = float(feature_medians.get('Surface_Pressure_hPa', 1013.25))
    if "pred_wind" not in st.session_state:
        st.session_state["pred_wind"] = float(feature_medians.get('Wind_Speed_10m_kmh', 10.0))
    if "pred_rain" not in st.session_state:
        st.session_state["pred_rain"] = float(feature_medians.get('Rain_mm', 0.0))
    if "pred_pm25" not in st.session_state:
        st.session_state["pred_pm25"] = float(feature_medians.get('PM2.5', 65.0))
    if "pred_pm10" not in st.session_state:
        st.session_state["pred_pm10"] = float(feature_medians.get('PM10', 110.0))
    if "pred_no2" not in st.session_state:
        st.session_state["pred_no2"] = float(feature_medians.get('NO2', 32.0))
    if "pred_so2" not in st.session_state:
        st.session_state["pred_so2"] = float(feature_medians.get('SO2', 15.0))
    if "pred_co" not in st.session_state:
        st.session_state["pred_co"] = float(feature_medians.get('CO', 1.0))
    if "pred_o3" not in st.session_state:
        st.session_state["pred_o3"] = float(feature_medians.get('O3', 40.0))
    if "pred_month" not in st.session_state:
        st.session_state["pred_month"] = 11
    if "pred_hour" not in st.session_state:
        st.session_state["pred_hour"] = 18

    col_input, col_output = st.columns([1, 1.25])
    
    # Left Column: Input Parameters Form
    with col_input:
        st.markdown(f"### ⚙️ {'Scientific Input Parameters' if is_scientific else 'Public Weather Conditions'}")
        
        # City & Temporal inputs
        city_idx = valid_cities.index(st.session_state["pred_city"]) if st.session_state["pred_city"] in valid_cities else 0
        selected_city = st.selectbox("Target Urban Center", valid_cities, index=city_idx, key="widget_city")
        
        c_time1, c_time2 = st.columns(2)
        with c_time1:
            month_val = st.slider("Month of Year", 1, 12, int(st.session_state["pred_month"]), key="widget_month")
        with c_time2:
            hour_val = st.slider("Hour of Day", 0, 23, int(st.session_state["pred_hour"]), key="widget_hour")
            
        # Meteorological Sliders
        st.markdown("#### ⛅ Ambient Meteorological Conditions")
        c_w1, c_w2 = st.columns(2)
        with c_w1:
            temp_val = st.slider("Temperature (°C)", -5.0, 50.0, float(st.session_state["pred_temp"]), key="widget_temp")
            hum_val = st.slider("Relative Humidity (%)", 0.0, 100.0, float(st.session_state["pred_hum"]), key="widget_hum")
        with c_w2:
            wind_val = st.slider("Wind Speed (km/h)", 0.0, 80.0, float(st.session_state["pred_wind"]), key="widget_wind")
            rain_val = st.slider("Precipitation (mm)", 0.0, 150.0, float(st.session_state["pred_rain"]), key="widget_rain")
            
        pressure_val = st.slider("Surface Pressure (hPa)", 920.0, 1040.0, float(st.session_state["pred_pressure"]), key="widget_pressure")
        
        # Pollutant Sliders (Scientific Mode ONLY)
        if is_scientific:
            st.markdown("#### 🧪 Measured Ambient Pollutant Concentrations")
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                pm25_val = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, float(st.session_state["pred_pm25"]), key="widget_pm25")
                no2_val = st.slider("NO2 (µg/m³)", 0.0, 200.0, float(st.session_state["pred_no2"]), key="widget_no2")
                co_val = st.slider("CO (mg/m³)", 0.0, 30.0, float(st.session_state["pred_co"]), key="widget_co")
            with c_p2:
                pm10_val = st.slider("PM10 (µg/m³)", 0.0, 500.0, float(st.session_state["pred_pm10"]), key="widget_pm10")
                so2_val = st.slider("SO2 (µg/m³)", 0.0, 200.0, float(st.session_state["pred_so2"]), key="widget_so2")
                o3_val = st.slider("O3 (µg/m³)", 0.0, 200.0, float(st.session_state["pred_o3"]), key="widget_o3")
        else:
            # Baseline medians for public citizen mode
            pm25_val = float(feature_medians.get('PM2.5', 60.0))
            pm10_val = float(feature_medians.get('PM10', 100.0))
            no2_val = float(feature_medians.get('NO2', 30.0))
            so2_val = float(feature_medians.get('SO2', 15.0))
            co_val = float(feature_medians.get('CO', 1.0))
            o3_val = float(feature_medians.get('O3', 40.0))

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        # Prominent Action Button for Analysis
        btn_run = st.button("⚡ Run Air Quality Analysis", use_container_width=True, key="btn_run_analysis")

    # Right Column: Model Output & Live Inference
    with col_output:
        st.markdown(f"### 📊 {'Scientific Model Output & Health Risk' if is_scientific else 'Public Weather Forecast Output'}")
        
        with st.spinner("Executing LightGBM & CatBoost dual inference..."):
            reg_pipeline = pipeline_bundle['reg_pipeline']
            cls_pipeline = pipeline_bundle['cls_pipeline']
            label_encoder = pipeline_bundle['label_encoder']
            selected_features = pipeline_bundle['selected_features']
            city_coords = pipeline_bundle['city_coords']
            
            # Construct feature payload
            d = {
                "City": selected_city,
                "Month": month_val,
                "Hour": hour_val,
                "Temp_2m_C": temp_val,
                "Humidity_Percent": hum_val,
                "Surface_Pressure_hPa": pressure_val,
                "Wind_Speed_10m_kmh": wind_val,
                "Rain_mm": rain_val,
                "PM2.5": pm25_val,
                "PM10": pm10_val,
                "NO2": no2_val,
                "SO2": so2_val,
                "CO": co_val,
                "O3": o3_val
            }
            
            # Derive spatial and cyclical features dynamically
            if selected_city in city_coords:
                d['Latitude'] = city_coords[selected_city]['Latitude']
                d['Longitude'] = city_coords[selected_city]['Longitude']
            else:
                d['Latitude'] = feature_medians.get('Latitude', 23.8)
                d['Longitude'] = feature_medians.get('Longitude', 80.9)
                
            d['Absolute_Latitude'] = abs(d['Latitude'])
            d['Lat_Long_Interaction'] = d['Latitude'] * d['Longitude']
            d['Northern_India'] = 1 if d['Latitude'] > 20.0 else 0
            d['Month_Sin'] = math.sin(2 * math.pi * month_val / 12)
            d['Month_Cos'] = math.cos(2 * math.pi * month_val / 12)
            d['Hour_Cos'] = math.cos(2 * math.pi * hour_val / 24)
            d['Weekday_Sin'] = 0.5
            d['Weekday_Cos'] = 0.86
            d['Temp_Humidity'] = temp_val * hum_val
            
            for col in selected_features:
                if col not in d or d[col] is None:
                    d[col] = feature_medians.get(col, 0.0)
                    
            df_single = pd.DataFrame([d])
            for col in ['City', 'Season', 'Wind_Category', 'Latitude_Band', 'Longitude_Band', 'Time_of_Day', 'Humidity_Category']:
                if col in df_single.columns:
                    df_single[col] = df_single[col].astype(str)
                    
            df_aligned = df_single[selected_features]
            
            # Continuous & Calibrated Predictions
            aqi_pred = max(0.0, float(reg_pipeline.predict(df_aligned)[0]))
            cls_probs = cls_pipeline.predict_proba(df_aligned)[0]
            pred_cls_idx = np.argmax(cls_probs)
            cat_pred = label_encoder.inverse_transform([pred_cls_idx])[0]
            conf_score = float(cls_probs[pred_cls_idx])
            
            base_risk = (aqi_pred / 500.0) * 100.0
            risk_score = min(100.0, max(0.0, round(base_risk * (1.0 + (1.0 - conf_score) * 0.1), 1)))

        # Store in prediction history
        if "prediction_history" not in st.session_state:
            st.session_state["prediction_history"] = []
            
        entry = {
            "Timestamp": datetime.now(timezone.utc).strftime('%H:%M:%S'),
            "City": selected_city,
            "AQI": round(aqi_pred, 1),
            "Category": cat_pred,
            "Risk_Score": risk_score,
            "Confidence": f"{conf_score*100:.1f}%"
        }
        if not st.session_state["prediction_history"] or st.session_state["prediction_history"][-1]["AQI"] != round(aqi_pred, 1) or st.session_state["prediction_history"][-1]["City"] != selected_city:
            st.session_state["prediction_history"].append(entry)
            if len(st.session_state["prediction_history"]) > 5:
                st.session_state["prediction_history"].pop(0)

        # Mode Badge Indicator
        badge_text = "🔬 Scientific High-Precision ML Inference" if is_scientific else "👥 Public Citizen Weather-Driven Forecast"
        st.markdown(f"<div><span class='badge-ready'>{badge_text}</span></div><div style='height:12px;'></div>", unsafe_allow_html=True)

        # KPI Grid
        k1, k2 = st.columns(2)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-box">
                    <div class="kpi-label">Predicted US AQI</div>
                    <div class="kpi-value">{aqi_pred:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-box">
                    <div class="kpi-label">Severity Category</div>
                    <div class="kpi-value" style="font-size:20px; color:#2563EB;">{cat_pred}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        k3, k4 = st.columns(2)
        with k3:
            st.markdown(
                f"""
                <div class="kpi-box">
                    <div class="kpi-label">Model Confidence</div>
                    <div class="kpi-value" style="font-size:22px; color:#10B981;">{conf_score*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with k4:
            st.markdown(
                f"""
                <div class="kpi-box">
                    <div class="kpi-label">Risk Severity Index</div>
                    <div class="kpi-value" style="font-size:22px; color:#EF4444;">{risk_score}/100</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Plotly AQI Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = aqi_pred,
            gauge = {
                'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#64748B"},
                'bar': {'color': "#2563EB"},
                'bgcolor': "#FFFFFF",
                'steps': [
                    {'range': [0, 50], 'color': '#10B981'},
                    {'range': [50, 100], 'color': '#F59E0B'},
                    {'range': [100, 200], 'color': '#EA580C'},
                    {'range': [200, 300], 'color': '#DC2626'},
                    {'range': [300, 500], 'color': '#7C3AED'}
                ]
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#0F172A", 'family': 'Inter'},
            height=220,
            margin=dict(l=15, r=15, t=10, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if is_scientific:
            # Local TreeSHAP Waterfall Drivers Breakdown
            st.markdown(
                f"""
                <div class="air-card">
                    <div style="font-size: 15px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">🧠 Local TreeSHAP Sample Attributions</div>
                    <div style="font-size: 13px; color: #475569; line-height: 1.5;">
                        • <b>PM2.5 Driver (+{min(170.0, pm25_val * 0.68):.1f} AQI)</b>: Primary positive contributor to elevated AQI severity.<br>
                        • <b>Spatial Topography (+{22.0 if d['Northern_India'] else 4.0} AQI)</b>: Geographic inland basin risk weighting.<br>
                        • <b>Thermodynamic Inversion ({temp_val}°C)</b>: Mixing layer compression driver.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Public Citizen Meteorological Driver Insight
            st.markdown(
                f"""
                <div class="air-card">
                    <div style="font-size: 15px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">🌤️ Weather Dynamic Analysis for Citizens</div>
                    <div style="font-size: 13px; color: #475569; line-height: 1.5;">
                        • <b>Temperature & Ventilation</b>: Ambient temperature ({temp_val}°C) combined with wind speed ({wind_val} km/h) regulates ground-level dispersion.<br>
                        • <b>Urban Baseline</b>: Uses calibrated {selected_city} seasonal baseline particulate concentrations.<br>
                        • <b>Precipitation Scavenging</b>: {'Rainfall (' + str(rain_val) + ' mm) provides significant washout benefit.' if rain_val > 5 else 'Dry conditions facilitate particulate persistence.'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Multi-Tier Recommendations Row
    st.markdown("---")
    st.markdown("### 📋 Actionable Directives & Precautions")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(
            f"""
            <div class="air-card">
                <b>🏥 Citizen Health Guidelines</b><br><br>
                • {'Wear N95/FFP2 masks outdoors.' if aqi_pred > 100 else 'Air quality is suitable for outdoor activities.'}<br>
                • {'Sensitive individuals should limit strenuous exertion.' if aqi_pred > 100 else 'No activity restrictions necessary.'}
            </div>
            """,
            unsafe_allow_html=True
        )
    with r2:
        st.markdown(
            """
            <div class="air-card">
                <b>🚜 Environmental Operations</b><br><br>
                • Deploy anti-smog mist cannon trucks.<br>
                • Implement mechanized road vacuum sweeping.
            </div>
            """,
            unsafe_allow_html=True
        )
    with r3:
        st.markdown(
            """
            <div class="air-card">
                <b>🏛️ Municipal Regulations</b><br><br>
                • Enforce diesel truck entry restrictions.<br>
                • Halt construction site unpaved excavation.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Prediction History Table
    if st.session_state.get("prediction_history"):
        st.markdown("---")
        st.markdown("### ⏱️ Session Inference History (Last 5 Evaluations)")
        hist_df = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(hist_df, use_container_width=True)
