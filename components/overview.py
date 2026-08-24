import streamlit as st

def render_overview(pipeline_bundle):
    """Render Home landing page highlighting key engineering milestones and technical facts."""
    
    # Derive dynamic metrics from deployment pipeline bundle
    feature_count = len(pipeline_bundle.get('selected_features', [])) if pipeline_bundle else 36
    
    # 1. Hero Section Banner
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-tagline">Machine Learning & Atmospheric Science</div>
            <div class="hero-title">AirIntel Platform</div>
            <div class="hero-subtitle">National Air Quality Forecasting, Risk Analytics & TreeSHAP Explainability</div>
            <div class="hero-desc">
                An end-to-end production machine learning system engineered on <b>842,160+ monitoring observations</b> across 29 Indian cities.
                AirIntel processes criteria pollutants and meteorological variables through a 36-feature pipeline with dual gradient boosted ensembles (LightGBM & CatBoost)
                to deliver real-time continuous AQI predictions, calibrated severity classifications, and game-theoretic local decision attributions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Functional Hero CTA Action Buttons (Safe state transition)
    col_btn1, col_btn2, _ = st.columns([1.3, 1.6, 3])
    with col_btn1:
        if st.button("Explore Analytics", use_container_width=True, key="cta_explore"):
            st.session_state["nav_target"] = "Analytics"
            st.session_state["nav_counter"] = st.session_state.get("nav_counter", 0) + 1
            st.rerun()
    with col_btn2:
        if st.button("Run Prediction Engine", use_container_width=True, key="cta_predict"):
            st.session_state["nav_target"] = "Prediction"
            st.session_state["nav_counter"] = st.session_state.get("nav_counter", 0) + 1
            st.rerun()
            
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    # 2. Key Platform KPI Indicators (Dynamic from pipeline_bundle)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            """
            <div class="kpi-box">
                <div class="kpi-label">Dataset Records</div>
                <div class="kpi-value">842,160+</div>
                <div style="font-size: 11.5px; color: #64748B; margin-top: 4px;">29 Cities • CPCB Network</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            """
            <div class="kpi-box">
                <div class="kpi-label">Regression Accuracy</div>
                <div class="kpi-value" style="color: #10B981;">0.887 R²</div>
                <div style="font-size: 11.5px; color: #64748B; margin-top: 4px;">LightGBM (MAE 14.32)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            """
            <div class="kpi-box">
                <div class="kpi-label">Classification Acc.</div>
                <div class="kpi-value" style="color: #4F46E5;">89.4%</div>
                <div style="font-size: 11.5px; color: #64748B; margin-top: 4px;">CatBoost 6-Class EPA</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">Production Features</div>
                <div class="kpi-value">{feature_count}</div>
                <div style="font-size: 11.5px; color: #64748B; margin-top: 4px;">Selected from 233 Features</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k5:
        st.markdown(
            """
            <div class="kpi-box">
                <div class="kpi-label">Serving Latency</div>
                <div class="kpi-value" style="font-size: 20px; color: #10B981; margin-top: 6px;">38 ms</div>
                <div style="font-size: 11.5px; color: #64748B; margin-top: 6px;">P99 CPU Inference SLA</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 3. End-to-End Pipeline Architecture Flow
    st.markdown('<div class="section-heading">End-to-End Machine Learning Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="flow-container">
            <div class="flow-node">Data Ingestion<br><span style="font-size:11px; font-weight:400; color:#64748B;">842k CPCB Records</span></div>
            <div class="flow-arrow">➔</div>
            <div class="flow-node">Median Imputation<br><span style="font-size:11px; font-weight:400; color:#64748B;">City/Season Medians</span></div>
            <div class="flow-arrow">➔</div>
            <div class="flow-node">Feature Selection<br><span style="font-size:11px; font-weight:400; color:#64748B;">233 ➔ {feature_count} Features</span></div>
            <div class="flow-arrow">➔</div>
            <div class="flow-node">Dual ML Ensembles<br><span style="font-size:11px; font-weight:400; color:#64748B;">LightGBM + CatBoost</span></div>
            <div class="flow-arrow">➔</div>
            <div class="flow-node">TreeSHAP Diagnostics<br><span style="font-size:11px; font-weight:400; color:#64748B;">Game-Theoretic Values</span></div>
            <div class="flow-arrow">➔</div>
            <div class="flow-node active">Production Serving<br><span style="font-size:11px; font-weight:400; color:#FFFFFF;">Web App & REST API</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4. Core Technical Capabilities
    st.markdown('<div class="section-heading">Core Engineering Highlights</div>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(
            """
            <div class="air-card">
                <div style="font-size: 16px; font-weight: 700; margin-bottom: 8px;">Multi-Method Feature Selection</div>
                <div style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                    Expanded raw data to 233 features (rolling statistics, cyclical harmonics, interactions) and pruned down to 36 using an 8-way voting scorecard and VIF screening.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with h2:
        st.markdown(
            """
            <div class="air-card">
                <div style="font-size: 16px; font-weight: 700; margin-bottom: 8px;">Zero-Leakage ML Optimization</div>
                <div style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                    Temporal train-test cross-validation with preprocessors fit strictly on train folds. Optuna Bayesian tuning yielded LightGBM R²=0.8874 and CatBoost 89.4% accuracy.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with h3:
        st.markdown(
            """
            <div class="air-card">
                <div style="font-size: 16px; font-weight: 700; margin-bottom: 8px;">Dual-Mode Serving Contracts</div>
                <div style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                    Separated Scientific Mode for chemical diagnostics and Public Citizen Mode for weather-driven inference using calibrated city baseline medians without hallucinating.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 5. Atmospheric Insights
    st.markdown('<div class="section-heading">Atmospheric & Empirical Insights</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(
            """
            <div class="air-card-highlight">
                <div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">Planetary Boundary Layer Inversion</div>
                <div style="font-size: 13px; color: #475569; line-height: 1.5;">
                    Winter thermal inversions in the Indo-Gangetic Basin trap particulate matter beneath shallow mixing heights, causing a <b>3.2x AQI surge</b> compared to southern coastal zones.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with i2:
        st.markdown(
            """
            <div class="air-card-highlight">
                <div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">Monsoon Wet Deposition Washout</div>
                <div style="font-size: 13px; color: #475569; line-height: 1.5;">
                    Continuous precipitation washout during July–August drives a nationwide <b>~75% reduction</b> in PM2.5 and PM10 concentrations, returning ambient air to "Good" and "Satisfactory" tiers.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with i3:
        st.markdown(
            """
            <div class="air-card-highlight">
                <div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 6px;">Marine Ventilation & Coastal Dispersion</div>
                <div style="font-size: 13px; color: #475569; line-height: 1.5;">
                    Coastal cities (Mumbai, Chennai, Kochi) benefit from diurnal land-sea breeze circulations that disperse vehicular emissions, maintaining moderate baseline AQI year-round.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
