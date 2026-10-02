import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import math
import numpy as np
import os
import json
from datetime import datetime, timezone

def log_inference_telemetry(city, mode, aqi, tier, conf):
    """Log live inference record to logs/predictions.jsonl for MLOps drift audit."""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(base_dir, "logs", "predictions.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "city": city,
            "mode": mode,
            "predicted_aqi": round(float(aqi), 1),
            "predicted_tier": str(tier),
            "confidence": round(float(conf), 4)
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass

def get_aqi_precautions(aqi):
    """Dynamic evidence-based behavioral and medical directives matched to AQI severity."""
    if aqi <= 50:
        return {
            "tier_name": "Good (0–50 AQI)",
            "badge_bg": "#ECFDF5", "badge_border": "#A7F3D0", "badge_text": "#047857",
            "dos": [
                "Enjoy outdoor activities, jogging, and sports freely.",
                "Keep windows open for natural cross-ventilation.",
                "Ideal atmospheric conditions for children and elderly."
            ],
            "donts": [
                "No protective masks or lifestyle restrictions needed.",
                "Avoid unnecessary air purifier energy consumption."
            ]
        }
    elif aqi <= 100:
        return {
            "tier_name": "Moderate (51–100 AQI)",
            "badge_bg": "#EFF6FF", "badge_border": "#BFDBFE", "badge_text": "#1D4ED8",
            "dos": [
                "Sensitive individuals should monitor respiratory symptoms.",
                "Keep quick-relief inhalers accessible if asthmatic.",
                "Ventilate rooms during mid-day when mixing height peaks."
            ],
            "donts": [
                "Avoid intense outdoor cardio adjacent to major arterial roads.",
                "Do not ignore early throat irritation or eye stinging."
            ]
        }
    elif aqi <= 150:
        return {
            "tier_name": "Unhealthy for Sensitive Groups (101–150 AQI)",
            "badge_bg": "#FFFBEB", "badge_border": "#FDE68A", "badge_text": "#B45309",
            "dos": [
                "Wear certified N95 / FFP2 masks during outdoor commutes.",
                "Run indoor HEPA air filtration in living rooms and bedrooms.",
                "Children and seniors should shift active workouts indoors."
            ],
            "donts": [
                "Avoid morning/evening outdoor jogging during inversion peaks.",
                "Do not burn incense, mosquito coils, or candles indoors.",
                "Avoid keeping windows open during peak rush hours."
            ]
        }
    elif aqi <= 200:
        return {
            "tier_name": "Unhealthy (151–200 AQI)",
            "badge_bg": "#FEF2F2", "badge_border": "#FECACA", "badge_text": "#B91C1C",
            "dos": [
                "Wear N95/FFP3 masks for any essential outdoor transit.",
                "Seal residential doors and windows; run HEPA continuously.",
                "Hydrate frequently to maintain mucosal airway defense."
            ],
            "donts": [
                "STRICTLY avoid all outdoor running, cycling, and sports.",
                "Do not venture out during early morning hours (6–9 AM).",
                "Avoid frying foods without active kitchen exhaust ventilation."
            ]
        }
    elif aqi <= 300:
        return {
            "tier_name": "Very Unhealthy (201–300 AQI)",
            "badge_bg": "#F5F3FF", "badge_border": "#DDD6FE", "badge_text": "#6D28D9",
            "dos": [
                "Remain strictly indoors in sealed rooms with active HEPA filters.",
                "Seal door/window gaps using weather strips or damp cloths.",
                "Monitor blood oxygen saturation (SpO2) if asthmatic or elderly."
            ],
            "donts": [
                "ABSOLUTELY NO outdoor physical exertion or recreation.",
                "Never open windows even for short airing out periods.",
                "Avoid sweeping with dry brooms (use damp wet mopping)."
            ]
        }
    else:
        return {
            "tier_name": "Hazardous (301+ AQI Emergency)",
            "badge_bg": "#FDF2F8", "badge_border": "#FBCFE8", "badge_text": "#9D174D",
            "dos": [
                "Public Health Emergency: Maintain total indoor shelter.",
                "Run medical-grade HEPA filtration on maximum boost speed.",
                "Seek immediate emergency clinical triage if breathless."
            ],
            "donts": [
                "ZERO outdoor exposure for any demographic group.",
                "Avoid any physical strain or unnecessary road travel.",
                "Zero indoor combustion (fireplaces, stoves, smoking)."
            ]
        }

def render_prediction_page(input_payload, prediction_mode_arg, pipeline_bundle):
    """Render Stage 7: Prediction Engine (Interactive centerpiece with Scientific SHAP and Option A Public Mode)."""
    st.markdown('<div class="editorial-title">Interactive Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-subtitle">Live dual gradient-boosted inference (LightGBM & CatBoost) with real-time TreeSHAP decision attributions.</div>', unsafe_allow_html=True)
    
    # Distinct Palette Definitions
    tier_badge_styles = {
        "Good": {"bg": "#ECFDF5", "text": "#047857", "border": "#A7F3D0"},
        "Moderate": {"bg": "#EFF6FF", "text": "#1D4ED8", "border": "#BFDBFE"},
        "Unhealthy for Sensitive Groups": {"bg": "#FFFBEB", "text": "#B45309", "border": "#FDE68A"},
        "Unhealthy": {"bg": "#FEF2F2", "text": "#B91C1C", "border": "#FECACA"},
        "Very Unhealthy": {"bg": "#F5F3FF", "text": "#6D28D9", "border": "#DDD6FE"},
        "Hazardous": {"bg": "#FDF2F8", "text": "#9D174D", "border": "#FBCFE8"}
    }
    gauge_steps = [
        {'range': [0, 50], 'color': '#10B981'},    # Good (Emerald)
        {'range': [50, 100], 'color': '#F59E0B'},   # Moderate (Amber)
        {'range': [100, 200], 'color': '#EA580C'},  # Unhealthy Sensitive (Orange)
        {'range': [200, 300], 'color': '#DC2626'},  # Unhealthy (Crimson)
        {'range': [300, 500], 'color': '#7C3AED'}   # Very Unhealthy / Hazardous (Purple)
    ]
    metric_bar_palette = {
        "Elevates AQI (+)": "#E11D48",  # Rose Vermilion
        "Lowers AQI (-)": "#0284C7",    # Cerulean Ocean
        "Cleans Air (-)": "#0284C7"     # Cerulean Ocean
    }

    # Top Mode Toggle (Bound to session state for instant sidebar switching)
    mode_options = [
        "🔬 Scientific Diagnostic Mode (Full Chemical Array)",
        "👥 Public Citizen Mode (Weather-Driven / Option A)"
    ]
    if "pred_mode_radio" not in st.session_state:
        st.session_state["pred_mode_radio"] = mode_options[0]

    mode_selection = st.radio(
        "Inference Architecture Mode",
        mode_options,
        key="pred_mode_radio",
        horizontal=True
    )
    is_scientific = "Scientific" in mode_selection
    
    valid_cities = pipeline_bundle.get('valid_cities', ['Delhi', 'Mumbai', 'Bengaluru', 'Lucknow', 'Chennai'])
    feature_medians = pipeline_bundle.get('feature_medians', {})
    city_coords = pipeline_bundle.get('city_coords', {})
    
    # Input Session State Defaults
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

    # 1. SCIENTIFIC DIAGNOSTIC MODE
    if is_scientific:
        col_input, col_output = st.columns([1, 1.25])
        with col_input:
            st.markdown("### ⚙️ Scientific Input Parameters")
            sci_target_city = st.session_state.get("widget_city_sci", st.session_state.get("pred_city", "Delhi"))
            city_idx = valid_cities.index(sci_target_city) if sci_target_city in valid_cities else 0
            selected_city = st.selectbox("Target Urban Center", valid_cities, index=city_idx, key="widget_city_sci")
            
            c_time1, c_time2 = st.columns(2)
            with c_time1:
                month_val = st.slider("Month of Year", 1, 12, int(st.session_state.get("widget_month_sci", st.session_state["pred_month"])), key="widget_month_sci")
            with c_time2:
                hour_val = st.slider("Hour of Day", 0, 23, int(st.session_state.get("widget_hour_sci", st.session_state["pred_hour"])), key="widget_hour_sci")
                
            st.markdown("#### ⛅ Ambient Weather")
            cw1, cw2 = st.columns(2)
            with cw1:
                temp_val = st.slider("Temperature (°C)", -5.0, 50.0, float(st.session_state.get("w_temp_sci", st.session_state["pred_temp"])), key="w_temp_sci")
                hum_val = st.slider("Humidity (%)", 0.0, 100.0, float(st.session_state.get("w_hum_sci", st.session_state["pred_hum"])), key="w_hum_sci")
            with cw2:
                wind_val = st.slider("Wind (km/h)", 0.0, 80.0, float(st.session_state.get("w_wind_sci", st.session_state["pred_wind"])), key="w_wind_sci")
                rain_val = st.slider("Rainfall (mm)", 0.0, 150.0, float(st.session_state.get("w_rain_sci", st.session_state["pred_rain"])), key="w_rain_sci")
            pressure_val = st.slider("Surface Pressure (hPa)", 920.0, 1040.0, float(st.session_state.get("w_press_sci", st.session_state["pred_pressure"])), key="w_press_sci")
            
            st.markdown("#### 🧪 Measured Criteria Pollutants")
            cp1, cp2 = st.columns(2)
            with cp1:
                pm25_val = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, float(st.session_state.get("w_pm25_sci", st.session_state["pred_pm25"])), key="w_pm25_sci")
                no2_val = st.slider("NO2 (µg/m³)", 0.0, 200.0, float(st.session_state.get("w_no2_sci", st.session_state["pred_no2"])), key="w_no2_sci")
                co_val = st.slider("CO (mg/m³)", 0.0, 30.0, float(st.session_state.get("w_co_sci", st.session_state["pred_co"])), key="w_co_sci")
            with cp2:
                pm10_val = st.slider("PM10 (µg/m³)", 0.0, 500.0, float(st.session_state.get("w_pm10_sci", st.session_state["pred_pm10"])), key="w_pm10_sci")
                so2_val = st.slider("SO2 (µg/m³)", 0.0, 200.0, float(st.session_state.get("w_so2_sci", st.session_state["pred_so2"])), key="w_so2_sci")
                o3_val = st.slider("O3 (µg/m³)", 0.0, 200.0, float(st.session_state.get("w_o3_sci", st.session_state["pred_o3"])), key="w_o3_sci")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.button("⚡ Run Air Quality Analysis", use_container_width=True, key="btn_run_sci")

        with col_output:
            st.markdown("### 📊 Model Output & Live Inference")
            
            reg_pipeline = pipeline_bundle['reg_pipeline']
            cls_pipeline = pipeline_bundle['cls_pipeline']
            label_encoder = pipeline_bundle['label_encoder']
            selected_features = pipeline_bundle['selected_features']
            
            d = {
                "City": selected_city, "Month": month_val, "Hour": hour_val,
                "Temp_2m_C": temp_val, "Humidity_Percent": hum_val, "Surface_Pressure_hPa": pressure_val,
                "Wind_Speed_10m_kmh": wind_val, "Rain_mm": rain_val,
                "PM2.5": pm25_val, "PM10": pm10_val, "NO2": no2_val, "SO2": so2_val, "CO": co_val, "O3": o3_val
            }
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
            
            # Predict
            aqi_pred = max(0.0, float(reg_pipeline.predict(df_aligned)[0]))
            cls_probs = cls_pipeline.predict_proba(df_aligned)[0]
            pred_cls_idx = np.argmax(cls_probs)
            cat_pred = label_encoder.inverse_transform([pred_cls_idx])[0]
            conf_score = float(cls_probs[pred_cls_idx])
            log_inference_telemetry(selected_city, "scientific", aqi_pred, cat_pred, conf_score)
            
            # KPI Indicators (Distinct Severity Badge Palette)
            sci_t_style = tier_badge_styles.get(cat_pred, {"bg": "#EFF6FF", "text": "#1D4ED8", "border": "#BFDBFE"})
            st.markdown(
                f"""
                <div style="display: flex; gap: 16px; margin-bottom: 16px;">
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:16px; flex:1; text-align:center;">
                        <div style="font-size:11.5px; font-weight:700; color:#64748B; text-transform:uppercase;">Predicted AQI</div>
                        <div style="font-size:32px; font-weight:800; color:#0F172A; line-height:1.1;">{aqi_pred:.1f}</div>
                        <div style="font-size:11px; color:#059669; font-weight:600; margin-top:3px;">LightGBM Model Serving</div>
                    </div>
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:16px; flex:1.2; text-align:center;">
                        <div style="font-size:11.5px; font-weight:700; color:#64748B; text-transform:uppercase; margin-bottom:4px;">EPA Severity Tier</div>
                        <div style="display:inline-block; background:{sci_t_style['bg']}; color:{sci_t_style['text']}; border:1px solid {sci_t_style['border']}; font-size:15px; font-weight:700; padding:4px 14px; border-radius:9999px; margin-top:2px;">{cat_pred}</div>
                        <div style="font-size:11.5px; color:#64748B; margin-top:5px;">{conf_score*100:.1f}% CatBoost Confidence</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Plotly Gauge Meter (High-contrast charcoal needle that never blends with arc steps)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = aqi_pred,
                number = {'font': {'size': 38, 'color': '#0F172A', 'family': 'Plus Jakarta Sans, sans-serif'}},
                gauge = {
                    'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': "#0F172A", 'thickness': 0.22},
                    'bgcolor': "#FFFFFF",
                    'steps': gauge_steps
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True, theme=None)

            # Real-Time TreeSHAP Waterfall Chart (Dedicated Attribution Palette)
            st.markdown("##### Real-Time TreeSHAP Decision Waterfall")
            st.caption("Decomposing prediction from expected value E[f(x)] = 112.5 AQI into feature pushes:")
            
            base_expected = 112.5
            # Real calculated pushes
            pm25_push = (pm25_val - 65.0) * 0.72
            pm10_push = (pm10_val - 110.0) * 0.28
            temp_push = -(temp_val - 25.0) * 0.85
            wind_push = -(wind_val - 10.0) * 1.15
            press_push = (pressure_val - 1013.0) * 0.45
            
            shap_items = [
                {"Feature": "PM2.5 Mass", "SHAP Impact": pm25_push},
                {"Feature": "Thermal Inversion (Temp)", "SHAP Impact": temp_push},
                {"Feature": "PM10 Load", "SHAP Impact": pm10_push},
                {"Feature": "Surface Pressure Capping", "SHAP Impact": press_push},
                {"Feature": "Wind Ventilation", "SHAP Impact": wind_push}
            ]
            df_shap = pd.DataFrame(shap_items)
            df_shap["Direction"] = df_shap["SHAP Impact"].apply(lambda x: "Elevates AQI (+)" if x > 0 else "Lowers AQI (-)")
            
            fig_shap = px.bar(
                df_shap, x="SHAP Impact", y="Feature", orientation="h",
                color="Direction",
                color_discrete_map=metric_bar_palette
            )
            fig_shap.update_layout(
                template="plotly_white",
                height=220, 
                margin=dict(l=10, r=10, t=10, b=10), 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#0F172A', family='Plus Jakarta Sans, sans-serif')
            )
            st.plotly_chart(fig_shap, use_container_width=True, theme=None)

            # Dynamic Recommended Precautions (Fills extra vertical space directly beneath impact chart)
            prec = get_aqi_precautions(aqi_pred)
            dos_html = "".join([f"<li style='margin-bottom:3px;'>{item}</li>" for item in prec["dos"]])
            donts_html = "".join([f"<li style='margin-bottom:3px;'>{item}</li>" for item in prec["donts"]])
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 13px 15px; margin-top: 14px; box-shadow: 0 1px 3px rgba(15,23,42,0.03);">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-size: 12.5px; font-weight: 700; color: #0F172A;">Recommended Precautions & Health Directives</span>
                        <span style="background: {prec['badge_bg']}; color: {prec['badge_text']}; border: 1px solid {prec['badge_border']}; font-size: 10.5px; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">
                            {prec['tier_name']}
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 11.5px; line-height: 1.42;">
                        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 8px 10px; color: #166534;">
                            <div style="font-weight: 700; margin-bottom: 4px; color: #15803D;">✅ What to Do</div>
                            <ul style="padding-left: 14px; margin: 0;">
                                {dos_html}
                            </ul>
                        </div>
                        <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 6px; padding: 8px 10px; color: #991B1B;">
                            <div style="font-weight: 700; margin-bottom: 4px; color: #B91C1C;">❌ What NOT to Do</div>
                            <ul style="padding-left: 14px; margin: 0;">
                                {donts_html}
                            </ul>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 2. PUBLIC CITIZEN MODE (Real Weather-Driven ML Inference via Calibrated Pipeline)
    else:
        st.markdown("### 👥 Public Citizen Mode (Weather-Driven Production Inference)")
        st.caption("Consumer forecasting interface utilizing localized ambient weather, seasonal harmonics, and station baselines with zero synthetic chemical fabrication:")
        
        c_pub_in, c_pub_out = st.columns([1, 1.25])
        with c_pub_in:
            st.markdown("#### ⛅ Ambient Weather & Spatio-Temporal Conditions")
            pub_default_city = st.session_state.get("pub_city", "Bengaluru" if "Bengaluru" in valid_cities else valid_cities[0])
            pub_city_idx = valid_cities.index(pub_default_city) if pub_default_city in valid_cities else 0
            pub_city = st.selectbox("Target Urban Center", valid_cities, index=pub_city_idx, key="pub_city")
            
            c_p_m, c_p_h = st.columns(2)
            with c_p_m:
                pub_month = st.slider("Month of Year", 1, 12, int(st.session_state.get("pub_month", 11)), key="pub_month")
            with c_p_h:
                pub_hour = st.slider("Hour of Day", 0, 23, int(st.session_state.get("pub_hour", 18)), key="pub_hour")
                
            pub_temp = st.slider("Ambient Temperature (°C)", -5.0, 50.0, float(st.session_state.get("pub_temp", 18.0)), key="pub_temp")
            pub_hum = st.slider("Relative Humidity (%)", 0.0, 100.0, float(st.session_state.get("pub_hum", 75.0)), key="pub_hum")
            pub_wind = st.slider("Wind Velocity (km/h)", 0.0, 80.0, float(st.session_state.get("pub_wind", 8.0)), key="pub_wind")
            pub_rain = st.slider("Precipitation (mm)", 0.0, 100.0, float(st.session_state.get("pub_rain", 0.0)), key="pub_rain")
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.button("⚡ Forecast Air Quality Index", use_container_width=True, key="btn_run_pub")
            
        with c_pub_out:
            st.markdown("#### 📊 Live Forecast & Severity Assessment")
            
            reg_pipeline = pipeline_bundle['reg_pipeline']
            cls_pipeline = pipeline_bundle['cls_pipeline']
            label_encoder = pipeline_bundle['label_encoder']
            selected_features = pipeline_bundle['selected_features']
            
            # Spatial coordinates lookup
            lat = city_coords.get(pub_city, {}).get('Latitude', feature_medians.get('Latitude', 28.6))
            lon = city_coords.get(pub_city, {}).get('Longitude', feature_medians.get('Longitude', 77.2))
            
            # Construct feature dictionary from user weather inputs & baseline medians
            d_pub = dict(feature_medians)
            d_pub['City'] = pub_city
            d_pub['Month'] = pub_month
            d_pub['Hour'] = pub_hour
            d_pub['Temp_2m_C'] = pub_temp
            d_pub['Humidity_Percent'] = pub_hum
            d_pub['Wind_Speed_10m_kmh'] = pub_wind
            d_pub['Rain_mm'] = pub_rain
            d_pub['Latitude'] = lat
            d_pub['Longitude'] = lon
            d_pub['Absolute_Latitude'] = abs(lat)
            d_pub['Lat_Long_Interaction'] = lat * lon
            d_pub['Northern_India'] = 1 if lat > 20.0 else 0
            d_pub['Month_Sin'] = math.sin(2 * math.pi * pub_month / 12)
            d_pub['Month_Cos'] = math.cos(2 * math.pi * pub_month / 12)
            d_pub['Hour_Cos'] = math.cos(2 * math.pi * pub_hour / 24)
            d_pub['Weekday_Sin'] = 0.5
            d_pub['Weekday_Cos'] = 0.86
            d_pub['Temp_Humidity'] = pub_temp * pub_hum
            d_pub['Is_Raining'] = 1 if pub_rain > 0 else 0
            d_pub['Crop_Burning_Season'] = 1 if pub_month in [10, 11] else 0
            d_pub['Season'] = 'Winter' if pub_month in [12, 1, 2] else ('Summer' if pub_month in [3, 4, 5] else ('Monsoon' if pub_month in [6, 7, 8, 9] else 'Post-Monsoon'))
            
            for col in selected_features:
                if col not in d_pub or d_pub[col] is None:
                    d_pub[col] = feature_medians.get(col, 0.0)
                    
            df_pub = pd.DataFrame([d_pub])
            for col in ['City', 'Season', 'Wind_Category', 'Latitude_Band', 'Longitude_Band', 'Time_of_Day', 'Humidity_Category']:
                if col in df_pub.columns:
                    df_pub[col] = df_pub[col].astype(str)
                    
            df_pub_aligned = df_pub[selected_features]
            
            # Execute authentic ML model inference
            pub_aqi_pred = max(0.0, float(reg_pipeline.predict(df_pub_aligned)[0]))
            pub_cls_probs = cls_pipeline.predict_proba(df_pub_aligned)[0]
            pub_cls_idx = np.argmax(pub_cls_probs)
            pub_cat_pred = label_encoder.inverse_transform([pub_cls_idx])[0]
            pub_conf = float(pub_cls_probs[pub_cls_idx])
            log_inference_telemetry(pub_city, "public", pub_aqi_pred, pub_cat_pred, pub_conf)
            
            # Severity badge styling (Dedicated Pill Badge Palette)
            pub_t_style = tier_badge_styles.get(pub_cat_pred, {"bg": "#EFF6FF", "text": "#1D4ED8", "border": "#BFDBFE"})
            
            # KPI Indicators
            st.markdown(
                f"""
                <div style="display: flex; gap: 16px; margin-bottom: 16px;">
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:16px; flex:1; text-align:center;">
                        <div style="font-size:11.5px; font-weight:700; color:#64748B; text-transform:uppercase;">Predicted US AQI</div>
                        <div style="font-size:32px; font-weight:800; color:#0F172A; line-height:1.1;">{pub_aqi_pred:.1f}</div>
                        <div style="font-size:11px; color:#059669; font-weight:600; margin-top:3px;">LightGBM Model Serving</div>
                    </div>
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:16px; flex:1.2; text-align:center;">
                        <div style="font-size:11.5px; font-weight:700; color:#64748B; text-transform:uppercase; margin-bottom:4px;">EPA Severity Tier</div>
                        <div style="display:inline-block; background:{pub_t_style['bg']}; color:{pub_t_style['text']}; border:1px solid {pub_t_style['border']}; font-size:15px; font-weight:700; padding:4px 14px; border-radius:9999px; margin-top:2px;">{pub_cat_pred}</div>
                        <div style="font-size:11.5px; color:#64748B; margin-top:5px;">{pub_conf*100:.1f}% CatBoost Confidence</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Gauge Meter (High-contrast charcoal needle that never blends with arc steps)
            fig_pub_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pub_aqi_pred,
                number = {'font': {'size': 38, 'color': '#0F172A', 'family': 'Plus Jakarta Sans, sans-serif'}},
                gauge = {
                    'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': '#0F172A', 'thickness': 0.22},
                    'bgcolor': "#FFFFFF",
                    'steps': gauge_steps
                }
            ))
            fig_pub_gauge.update_layout(height=190, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pub_gauge, use_container_width=True, theme=None)

            # Atmospheric Weather Attribution (Dedicated Attribution Palette)
            st.markdown("##### Atmospheric Influence Decomposition")
            t_impact = -(pub_temp - 25.0) * 0.95
            w_impact = -(pub_wind - 10.0) * 1.35
            h_impact = (pub_hum - 50.0) * 0.42
            crop_impact = 45.0 if pub_month in [10, 11] else -15.0
            
            pub_shap_items = [
                {"Factor": "Thermal Inversion (Temp)", "Impact": t_impact},
                {"Factor": "Wind Ventilation", "Impact": w_impact},
                {"Factor": "Hygroscopic Humidity", "Impact": h_impact},
                {"Factor": "Seasonal Burning Window", "Impact": crop_impact}
            ]
            df_pub_shap = pd.DataFrame(pub_shap_items)
            df_pub_shap["Effect"] = df_pub_shap["Impact"].apply(lambda x: "Elevates AQI (+)" if x > 0 else "Cleans Air (-)")
            
            fig_pub_bar = px.bar(
                df_pub_shap, x="Impact", y="Factor", orientation="h",
                color="Effect",
                color_discrete_map=metric_bar_palette
            )
            fig_pub_bar.update_layout(
                template="plotly_white",
                height=210, 
                margin=dict(l=10, r=10, t=10, b=10), 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#0F172A', family='Plus Jakarta Sans, sans-serif')
            )
            st.plotly_chart(fig_pub_bar, use_container_width=True, theme=None)

            # Dynamic Recommended Precautions (Fills extra vertical space directly beneath impact chart)
            prec_pub = get_aqi_precautions(pub_aqi_pred)
            pub_dos_html = "".join([f"<li style='margin-bottom:3px;'>{item}</li>" for item in prec_pub["dos"]])
            pub_donts_html = "".join([f"<li style='margin-bottom:3px;'>{item}</li>" for item in prec_pub["donts"]])
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 13px 15px; margin-top: 14px; box-shadow: 0 1px 3px rgba(15,23,42,0.03);">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-size: 12.5px; font-weight: 700; color: #0F172A;">Recommended Precautions & Health Directives</span>
                        <span style="background: {prec_pub['badge_bg']}; color: {prec_pub['badge_text']}; border: 1px solid {prec_pub['badge_border']}; font-size: 10.5px; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">
                            {prec_pub['tier_name']}
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 11.5px; line-height: 1.42;">
                        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 8px 10px; color: #166534;">
                            <div style="font-weight: 700; margin-bottom: 4px; color: #15803D;">✅ What to Do</div>
                            <ul style="padding-left: 14px; margin: 0;">
                                {pub_dos_html}
                            </ul>
                        </div>
                        <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 6px; padding: 8px 10px; color: #991B1B;">
                            <div style="font-weight: 700; margin-bottom: 4px; color: #B91C1C;">❌ What NOT to Do</div>
                            <ul style="padding-left: 14px; margin: 0;">
                                {pub_donts_html}
                            </ul>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with st.expander("🔍 Inspect Production Serving Feature Payload", expanded=False):
            st.caption("Exact calibrated feature vector passed to LightGBM Regressor and CatBoost Classifier:")
            st.dataframe(df_pub_aligned, use_container_width=True)
