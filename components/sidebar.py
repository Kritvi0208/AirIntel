import streamlit as st
import pandas as pd

def render_sidebar(selected_page, valid_cities, feature_medians):
    """Render ONE rich, dedicated context-specific sidebar tailored exactly to the active tab."""
    sidebar_filters = {}
    prediction_mode = "scientific"
    
    with st.sidebar:
        # Platform Brand Header
        st.markdown(
            """
            <div style="padding: 4px 0 10px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="font-size: 20px; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">AirIntel</div>
                    <span style="background: #ECFDF5; color: #059669; font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 9999px; border: 1px solid #A7F3D0;">● LIVE</span>
                </div>
                <div style="font-size: 11px; font-weight: 700; color: #2563EB; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px;">National Atmospheric Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # -------------------------------------------------------------
        # TAB 1: OVERVIEW SIDEBAR
        # -------------------------------------------------------------
        if selected_page == "Overview":
            st.markdown("#### 🏠 Overview Summary")
            st.markdown(
                """
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px; margin-bottom:12px; font-size:12.5px; line-height:1.5;">
                    <div style="font-weight:700; color:#0F172A; margin-bottom:4px;">Dataset Benchmark Scale</div>
                    • <b>842,160+</b> Hourly CAAQMS Records<br>
                    • <b>29</b> Major Urban Monitoring Centers<br>
                    • <b>Multi-Year Longitudinal</b> Grid<br>
                    • <b>6</b> Criteria Atmospheric Pollutants
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("#### 🏆 Champion Models")
            st.markdown(
                """
                <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:6px; padding:10px 12px; margin-bottom:14px; font-size:12.5px; line-height:1.5; color:#1E40AF;">
                    • <b>LightGBM Regressor</b>: R² = 0.8874<br>
                    • <b>CatBoost Classifier</b>: 89.4% Accuracy<br>
                    • <b>Inference Latency</b>: 18.7 ms CPU
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("#### ⚡ Quick Navigation")
            if st.button("🔮 Launch Prediction Engine", use_container_width=True, key="sb_btn_pred"):
                st.session_state["active_page"] = "Prediction"
                st.session_state["nav_counter"] = st.session_state.get("nav_counter", 0) + 1
                st.rerun()
            if st.button("📊 Explore EDA & Trends", use_container_width=True, key="sb_btn_eda"):
                st.session_state["active_page"] = "EDA & Statistics"
                st.session_state["nav_counter"] = st.session_state.get("nav_counter", 0) + 1
                st.rerun()
            if st.button("📈 View Model Leaderboards", use_container_width=True, key="sb_btn_model"):
                st.session_state["active_page"] = "Modeling"
                st.session_state["nav_counter"] = st.session_state.get("nav_counter", 0) + 1
                st.rerun()

        # -------------------------------------------------------------
        # TAB 2: RESEARCH JOURNEY SIDEBAR
        # -------------------------------------------------------------
        elif selected_page == "Research Journey":
            st.markdown("#### 🗺️ Pipeline Roadmap")
            st.markdown(
                """
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px; margin-bottom:12px; font-size:12.5px; line-height:1.5;">
                    <div style="font-weight:700; color:#0F172A; margin-bottom:4px;">14-Stage Architecture</div>
                    • <b>Phase 1</b>: Ingestion (01–03)<br>
                    • <b>Phase 2</b>: Discovery (04–06)<br>
                    • <b>Phase 3</b>: ML & Tuning (07–09)<br>
                    • <b>Phase 4</b>: Deployment & MLOps (10–14)
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("#### 💡 Key Innovations")
            st.markdown(
                """
                <div style="font-size:12px; color:#334155; line-height:1.5; margin-bottom:14px;">
                    • <b>>35% Missingness Rescued</b> via City × Season Median Imputation.<br>
                    • <b>233 Domain Features</b> using Cyclical Harmonics & Thermodynamic Aerosol physics.<br>
                    • <b>8-Way Consensus</b> feature selection scorecard.
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.caption("Inspect each notebook step using the interactive flowchart board on the main canvas.")

        # -------------------------------------------------------------
        # TAB 3: EDA & STATISTICS SIDEBAR
        # -------------------------------------------------------------
        elif selected_page == "EDA & Statistics":
            st.markdown("#### 📊 Exploratory Filters")
            
            city_choice = st.selectbox(
                "Filter Urban Station",
                ["All 29 Cities (National Baseline)"] + valid_cities,
                index=0,
                key="sb_eda_city"
            )
            sidebar_filters["city"] = city_choice
            

            
            season_choice = st.selectbox(
                "Atmospheric Season",
                ["All Seasons", "Winter (Inversion)", "Monsoon (Scavenging)", "Summer (Heatwave)", "Post-Monsoon (Burning)"],
                index=0,
                key="sb_eda_season"
            )
            sidebar_filters["season"] = season_choice
            
            st.markdown("#### 🔬 Atmospheric Rules")
            st.markdown(
                """
                <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:6px; padding:8px 10px; font-size:12px; color:#1E40AF; line-height:1.45;">
                    • <b>PM2.5 Dominance</b>: r = 0.92 correlation with US AQI.<br>
                    • <b>Rain Washout</b>: ~75% particulate drop in July-August.<br>
                    • <b>Winter Inversion</b>: 3.2× pollution surge in Gangetic basin.
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------------------
        # TAB 4: MODELING SIDEBAR
        # -------------------------------------------------------------
        elif selected_page == "Modeling":
            st.markdown("#### 📈 Benchmark Controls")
            family_choice = st.selectbox(
                "Filter Model Family",
                ["All 11 Models Tested", "Gradient Boosted Trees (LightGBM, CatBoost, XGBoost)", "Ensemble Bagging (Random Forest, Extra Trees)", "Linear & Regularized Baselines"],
                index=0,
                key="sb_mod_family"
            )
            sidebar_filters["family"] = family_choice
            
            st.markdown("#### 🔒 Zero Data Leakage")
            st.markdown(
                """
                <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:6px; padding:10px 12px; margin-bottom:12px; font-size:12px; color:#065F46; line-height:1.5;">
                    • <b>Strict Temporal Split</b>: 80% train, 20% test chronologically.<br>
                    • <b>Transformer Isolation</b>: StandardScaler and OneHotEncoder fitted strictly on training folds.<br>
                    • <b>5-Fold CV</b>: Cross-validated across time series slices.
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("#### ⚠️ Rejection Note")
            st.caption("Random Forest achieved Train R² = 0.981 but was rejected due to Test R² = 0.852 overfitting and an unmanageable 448 MB memory footprint.")

        # -------------------------------------------------------------
        # TAB 5: OPTIMIZATION SIDEBAR
        # -------------------------------------------------------------
        elif selected_page == "Optimization":
            st.markdown("#### ⚡ Pruning & Tuning Controls")
            min_consensus = st.slider(
                "Consensus Scorecard Threshold",
                min_value=1,
                max_value=8,
                value=6,
                help="Minimum number of voting algorithms required for inclusion",
                key="sb_opt_votes"
            )
            sidebar_filters["min_votes"] = min_consensus
            
            st.markdown("#### 🎯 Optuna Hyperband Vitals")
            st.markdown(
                """
                <div style="background:#FAF5FF; border:1px solid #E9D5FF; border-radius:6px; padding:10px 12px; margin-bottom:12px; font-size:12px; color:#6B21A8; line-height:1.5;">
                    • <b>Trials Evaluated</b>: 100 Bayesian Trials<br>
                    • <b>Pruning Strategy</b>: Median Trial Pruner<br>
                    • <b>RMSE Gain</b>: <b>40.4% Reduction</b><br>
                    • <b>Validation RMSE</b>: 31.17 ➔ 18.57
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption("Pruned 233 candidate features down to 36 production features.")

        # -------------------------------------------------------------
        # TAB 6: ADVANCED ANALYTICS SIDEBAR
        # -------------------------------------------------------------
        elif selected_page == "Advanced Analytics":
            st.markdown("#### 🔬 Spatial Intelligence Controls")
            cluster_filter = st.radio(
                "City Cluster Archetype",
                ["Both Clusters (29 Cities)", "Cluster 0: Clean / Coastal (12)", "Cluster 1: Industrial / Gangetic (17)"],
                key="sb_ana_cluster"
            )
            sidebar_filters["cluster"] = cluster_filter
            
            sidebar_filters["projection"] = "PCA (Linear Variance)"
            
            st.markdown("#### 🌉 Network Bridges")
            st.markdown(
                """
                <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:6px; padding:10px 12px; font-size:12px; color:#92400E; line-height:1.5;">
                    • <b>Agartala</b>: Betweenness = 0.259 (Northeast bridge)<br>
                    • <b>Raipur</b>: Betweenness = 0.178 (Central corridor bridge)
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------------------
        # TAB 7: PREDICTION ENGINE SIDEBAR
        # -------------------------------------------------------------
        elif selected_page == "Prediction":
            st.markdown("#### 🔬 Scientific Quick Presets")
            st.caption("3 Distinct Industrial & Thermal Basins:")
            
            # City 1: Delhi
            if st.button("📍 Delhi (Winter Inversion)", use_container_width=True, key="preset_sci_delhi"):
                city_val = "Delhi" if "Delhi" in valid_cities else valid_cities[0]
                st.session_state["pred_mode_radio"] = "🔬 Scientific Diagnostic Mode (Full Chemical Array)"
                # Set active widget keys
                st.session_state["widget_city_sci"] = city_val
                st.session_state["widget_month_sci"] = 12
                st.session_state["widget_hour_sci"] = 21
                st.session_state["w_temp_sci"] = 11.5
                st.session_state["w_hum_sci"] = 84.0
                st.session_state["w_press_sci"] = 1018.0
                st.session_state["w_wind_sci"] = 4.2
                st.session_state["w_rain_sci"] = 0.0
                st.session_state["w_pm25_sci"] = 285.0
                st.session_state["w_pm10_sci"] = 420.0
                st.session_state["w_no2_sci"] = 68.0
                st.session_state["w_so2_sci"] = 24.0
                st.session_state["w_co_sci"] = 2.8
                st.session_state["w_o3_sci"] = 32.0
                # Synchronize internal session state keys
                st.session_state["pred_city"] = city_val
                st.session_state["pred_month"] = 12
                st.session_state["pred_hour"] = 21
                st.session_state["pred_temp"] = 11.5
                st.session_state["pred_hum"] = 84.0
                st.session_state["pred_pressure"] = 1018.0
                st.session_state["pred_wind"] = 4.2
                st.session_state["pred_rain"] = 0.0
                st.session_state["pred_pm25"] = 285.0
                st.session_state["pred_pm10"] = 420.0
                st.session_state["pred_no2"] = 68.0
                st.session_state["pred_so2"] = 24.0
                st.session_state["pred_co"] = 2.8
                st.session_state["pred_o3"] = 32.0
                st.rerun()

            # City 2: Kolkata
            if st.button("📍 Kolkata (Gangetic Smog)", use_container_width=True, key="preset_sci_kolkata"):
                city_val = "Kolkata" if "Kolkata" in valid_cities else valid_cities[0]
                st.session_state["pred_mode_radio"] = "🔬 Scientific Diagnostic Mode (Full Chemical Array)"
                # Set active widget keys
                st.session_state["widget_city_sci"] = city_val
                st.session_state["widget_month_sci"] = 1
                st.session_state["widget_hour_sci"] = 9
                st.session_state["w_temp_sci"] = 18.0
                st.session_state["w_hum_sci"] = 76.0
                st.session_state["w_press_sci"] = 1014.0
                st.session_state["w_wind_sci"] = 6.5
                st.session_state["w_rain_sci"] = 0.0
                st.session_state["w_pm25_sci"] = 160.0
                st.session_state["w_pm10_sci"] = 240.0
                st.session_state["w_no2_sci"] = 52.0
                st.session_state["w_so2_sci"] = 18.0
                st.session_state["w_co_sci"] = 1.9
                st.session_state["w_o3_sci"] = 45.0
                # Synchronize internal session state keys
                st.session_state["pred_city"] = city_val
                st.session_state["pred_month"] = 1
                st.session_state["pred_hour"] = 9
                st.session_state["pred_temp"] = 18.0
                st.session_state["pred_hum"] = 76.0
                st.session_state["pred_pressure"] = 1014.0
                st.session_state["pred_wind"] = 6.5
                st.session_state["pred_rain"] = 0.0
                st.session_state["pred_pm25"] = 160.0
                st.session_state["pred_pm10"] = 240.0
                st.session_state["pred_no2"] = 52.0
                st.session_state["pred_so2"] = 18.0
                st.session_state["pred_co"] = 1.9
                st.session_state["pred_o3"] = 45.0
                st.rerun()

            # City 3: Patna
            if st.button("📍 Patna (Basin Stagnation)", use_container_width=True, key="preset_sci_patna"):
                city_val = "Patna" if "Patna" in valid_cities else valid_cities[0]
                st.session_state["pred_mode_radio"] = "🔬 Scientific Diagnostic Mode (Full Chemical Array)"
                # Set active widget keys
                st.session_state["widget_city_sci"] = city_val
                st.session_state["widget_month_sci"] = 11
                st.session_state["widget_hour_sci"] = 20
                st.session_state["w_temp_sci"] = 16.5
                st.session_state["w_hum_sci"] = 82.0
                st.session_state["w_press_sci"] = 1015.5
                st.session_state["w_wind_sci"] = 3.8
                st.session_state["w_rain_sci"] = 0.0
                st.session_state["w_pm25_sci"] = 225.0
                st.session_state["w_pm10_sci"] = 340.0
                st.session_state["w_no2_sci"] = 58.0
                st.session_state["w_so2_sci"] = 21.0
                st.session_state["w_co_sci"] = 2.4
                st.session_state["w_o3_sci"] = 38.0
                # Synchronize internal session state keys
                st.session_state["pred_city"] = city_val
                st.session_state["pred_month"] = 11
                st.session_state["pred_hour"] = 20
                st.session_state["pred_temp"] = 16.5
                st.session_state["pred_hum"] = 82.0
                st.session_state["pred_pressure"] = 1015.5
                st.session_state["pred_wind"] = 3.8
                st.session_state["pred_rain"] = 0.0
                st.session_state["pred_pm25"] = 225.0
                st.session_state["pred_pm10"] = 340.0
                st.session_state["pred_no2"] = 58.0
                st.session_state["pred_so2"] = 21.0
                st.session_state["pred_co"] = 2.4
                st.session_state["pred_o3"] = 38.0
                st.rerun()

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.markdown("#### 👥 Public Mode Presets")
            st.caption("3 Distinct Weather-Driven Cities:")

            # City 4: Bengaluru
            if st.button("📍 Bengaluru (Clean Plateau)", use_container_width=True, key="preset_pub_bengaluru"):
                city_val = "Bengaluru" if "Bengaluru" in valid_cities else valid_cities[0]
                st.session_state["pred_mode_radio"] = "👥 Public Citizen Mode (Weather-Driven / Option A)"
                # Set public mode widget keys
                st.session_state["pub_city"] = city_val
                st.session_state["pub_month"] = 7
                st.session_state["pub_hour"] = 10
                st.session_state["pub_temp"] = 22.0
                st.session_state["pub_hum"] = 58.0
                st.session_state["pub_wind"] = 14.5
                st.session_state["pub_rain"] = 0.0
                # Synchronize scientific keys as well for seamless switching
                st.session_state["widget_city_sci"] = city_val
                st.session_state["pred_city"] = city_val
                st.session_state["pred_month"] = 7
                st.session_state["pred_hour"] = 10
                st.session_state["pred_temp"] = 22.0
                st.session_state["pred_hum"] = 58.0
                st.session_state["pred_wind"] = 14.5
                st.session_state["pred_rain"] = 0.0
                st.rerun()

            # City 5: Mumbai
            if st.button("📍 Mumbai (Monsoon Washout)", use_container_width=True, key="preset_pub_mumbai"):
                city_val = "Mumbai" if "Mumbai" in valid_cities else valid_cities[0]
                st.session_state["pred_mode_radio"] = "👥 Public Citizen Mode (Weather-Driven / Option A)"
                # Set public mode widget keys
                st.session_state["pub_city"] = city_val
                st.session_state["pub_month"] = 8
                st.session_state["pub_hour"] = 14
                st.session_state["pub_temp"] = 27.5
                st.session_state["pub_hum"] = 92.0
                st.session_state["pub_wind"] = 24.0
                st.session_state["pub_rain"] = 42.0
                # Synchronize scientific keys as well for seamless switching
                st.session_state["widget_city_sci"] = city_val
                st.session_state["pred_city"] = city_val
                st.session_state["pred_month"] = 8
                st.session_state["pred_hour"] = 14
                st.session_state["pred_temp"] = 27.5
                st.session_state["pred_hum"] = 92.0
                st.session_state["pred_wind"] = 24.0
                st.session_state["pred_rain"] = 42.0
                st.rerun()

            # City 6: Chennai
            if st.button("📍 Chennai (Coastal Breeze)", use_container_width=True, key="preset_pub_chennai"):
                city_val = "Chennai" if "Chennai" in valid_cities else valid_cities[0]
                st.session_state["pred_mode_radio"] = "👥 Public Citizen Mode (Weather-Driven / Option A)"
                # Set public mode widget keys
                st.session_state["pub_city"] = city_val
                st.session_state["pub_month"] = 10
                st.session_state["pub_hour"] = 16
                st.session_state["pub_temp"] = 31.0
                st.session_state["pub_hum"] = 85.0
                st.session_state["pub_wind"] = 18.0
                st.session_state["pub_rain"] = 6.0
                # Synchronize scientific keys as well for seamless switching
                st.session_state["widget_city_sci"] = city_val
                st.session_state["pred_city"] = city_val
                st.session_state["pred_month"] = 10
                st.session_state["pred_hour"] = 16
                st.session_state["pred_temp"] = 31.0
                st.session_state["pred_hum"] = 85.0
                st.session_state["pred_wind"] = 18.0
                st.session_state["pred_rain"] = 6.0
                st.rerun()

            st.markdown(
                """
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 12px; margin-top:14px; font-size:12px; color:#475569; line-height:1.5;">
                    💡 <b>Tip:</b> Click any preset above to test 6 distinct geographic corridors across Scientific and Public modes with 1-click calibration.
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------------------
        # TAB 8: EXPLAINABILITY & SPATIAL SIDEBAR
        # -------------------------------------------------------------
        elif selected_page == "Explainability & Spatial":
            st.markdown("#### 🌍 Spatial & Diagnostic Controls")
            metric_map = st.selectbox(
                "Station Map Bubble Metric",
                ["Average US AQI", "Average PM2.5 (µg/m³)", "Model Test MAE", "AQI Median"],
                index=0,
                key="sb_exp_metric"
            )
            sidebar_filters["map_metric"] = metric_map
            
            st.markdown("#### 🔍 Hotspot Diagnostics")
            st.markdown(
                """
                <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:6px; padding:10px 12px; margin-bottom:12px; font-size:12px; color:#92400E; line-height:1.5;">
                    • <b>Peak Variance Hotspot</b>: Gurugram (MAE = 20.94) & Delhi (MAE = 19.55).<br>
                    • <b>Clean Low-Error Cities</b>: Kohima (MAE = 8.92) & Itanagar (MAE = 8.93).
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.caption("Map uses clean OpenStreetMap tiles with zero API keys or vendor watermarks.")
        # -------------------------------------------------------------
        # TAB 9: ABOUT SIDEBAR
        # -------------------------------------------------------------
        elif selected_page == "About":
            st.markdown("#### ℹ️ Project Provenance")
            st.markdown(
                """
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px; margin-bottom:12px; font-size:12.5px; line-height:1.5;">
                    • <b>Author</b>: Ritvika<br>
                    • <b>Institution</b>: Indian Institute of Technology (BHU)<br>
                    • <b>Data Source</b>: CPCB CAAQMS Grid (842k+ Records)<br>
                    • <b>Repository</b>: Open Research
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("#### 📦 Stack")
            st.caption("Python 3.13 • Streamlit • LightGBM • CatBoost • SHAP • Plotly • Scikit-Learn")

        # Static clean footer without absolute positioning
        st.markdown(
            """
            <div style="margin-top: 24px; padding-top: 12px; border-top: 1px solid #E2E8F0; font-size: 11px; color: #94A3B8; text-align: center;">
                AirIntel v2.0 • Scientific Edition
            </div>
            """,
            unsafe_allow_html=True
        )

    return sidebar_filters, prediction_mode
