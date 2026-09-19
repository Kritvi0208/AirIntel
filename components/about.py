import streamlit as st

def render_about(pipeline_bundle):
    """Render Stage 10: About (Prestigious IIT BHU Academic Showcase & Project Provenance)."""
    
    # 1. INSTITUTIONAL HEADER BANNER
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 50%, #EFF6FF 100%); border: 1px solid #DBEAFE; border-radius: 16px; padding: 32px 36px; box-shadow: 0 2px 10px rgba(37, 99, 235, 0.04); margin-bottom: 24px;">
            <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 10px;">
                <span style="background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 9999px; letter-spacing: 0.05em;">
                    NATIONAL AIR QUALITY INTELLIGENCE PLATFORM • v2.0
                </span>
                <span style="background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 9999px;">
                    PRODUCTION SERVING ACTIVE
                </span>
            </div>
            <div style="font-size: 32px; font-weight: 800; color: #0F172A; letter-spacing: -0.03em; margin: 4px 0 8px 0; line-height: 1.15;">
                AirIntel: Atmospheric Data Science & Machine Learning Platform
            </div>
            <div style="font-size: 14.5px; color: #475569; line-height: 1.6; max-width: 960px; margin-bottom: 16px;">
                A research initiative synthesizing 842,160+ hourly observations from India's Central Pollution Control Board (CPCB) continuous monitoring network. AirIntel bridges atmospheric physics with production-grade gradient boosting to deliver continuous US AQI forecasting, 6-class EPA severity triage, and real-time TreeSHAP decision explanations under a verified Zero Data Leakage scientific protocol.
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; font-weight: 600; color: #334155;">
                <span style="background: #FFFFFF; padding: 5px 12px; border-radius: 8px; border: 1px solid #E2E8F0;">Academic Research: Ritvika</span>
                <span style="background: #FFFFFF; padding: 5px 12px; border-radius: 8px; border: 1px solid #E2E8F0;">CPCB CAAQMS Benchmark (842,160+ Records)</span>
                <span style="background: #FFFFFF; padding: 5px 12px; border-radius: 8px; border: 1px solid #E2E8F0;">LightGBM R²=0.8874 • CatBoost 89.4%</span>
                <span style="background: #FFFFFF; padding: 5px 12px; border-radius: 8px; border: 1px solid #E2E8F0;">MIT Open Source License</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # 2. THREE CORE SCIENTIFIC PILLARS
    st.markdown('<div class="section-heading">Foundational Research Pillars</div>', unsafe_allow_html=True)
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown(
            """
            <div class="air-card">
                <div style="font-size: 16px; font-weight: 800; color: #0F172A; margin-bottom: 6px;">Atmospheric Physics Modeling</div>
                <div style="font-size: 13px; color: #475569; line-height: 1.55;">
                    Modeled the <b>Planetary Boundary Layer (PBL) thermal inversions</b> that compress winter pollutants in the landlocked Indo-Gangetic Basin (causing 3.2× AQI surges). Quantified monsoon rain scavenging (~75% particulate washout) and cyclical solar diurnal peaks.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_p2:
        st.markdown(
            """
            <div class="air-card">
                <div style="font-size: 16px; font-weight: 800; color: #0F172A; margin-bottom: 6px;">Zero-Leakage ML Engineering</div>
                <div style="font-size: 13px; color: #475569; line-height: 1.55;">
                    Engineered 233 features from thermodynamics, harmonic sine/cosine cycles, and moving averages. Pruned down to 36 production features using an <b>8-way consensus scorecard</b>, tuned with 100 Optuna Bayesian trials for a <b>40.4% RMSE reduction</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_p3:
        st.markdown(
            """
            <div class="air-card">
                <div style="font-size: 16px; font-weight: 800; color: #0F172A; margin-bottom: 6px;">Explainability & SLA Serving</div>
                <div style="font-size: 13px; color: #475569; line-height: 1.55;">
                    Integrated <b>TreeSHAP game-theoretic attribution</b> to provide local additive feature pushes for every prediction. Encapsulated pipelines into a 3.57 MB asset achieving <b>18.7 ms inference latency</b> (P99 SLA < 38 ms) on commodity CPUs.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # 3. METHODOLOGY & DATA ARCHITECTURE
    col_meth, col_side = st.columns([1.6, 1])
    with col_meth:
        st.markdown('<div class="section-heading">Dataset Provenance & Preparation Protocol</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:20px 24px; font-size:13.5px; color:#334155; line-height:1.65; margin-bottom:20px;">
                <b>Central Pollution Control Board (CPCB) CAAQMS Telemetry:</b><br>
                The research dataset comprises 842,160+ continuous hourly observations across 29 major Indian urban centers synthesized across extensive multi-year longitudinal monitoring. 
                Monitored parameters include:
                <ul style="margin: 8px 0; padding-left: 20px;">
                    <li><b>Criteria Particulates:</b> PM2.5 (Fine particulate mass) and PM10 (Coarse particulate matter).</li>
                    <li><b>Gaseous Pollutants:</b> Nitrogen Dioxide (NO2), Sulfur Dioxide (SO2), Carbon Monoxide (CO), Ozone (O3).</li>
                    <li><b>Meteorological Channels:</b> Surface Temperature, Relative Humidity, Surface Pressure, Wind Speed, Wind Direction, Precipitation.</li>
                </ul>
                <b>Variance-Preserving Data Imputation:</b><br>
                Initial auditing identified >35% missingness in long-term sensor telemetry. Naive mean imputation was rejected because it distorts localized microclimates (e.g. imputing Delhi winter values into Mumbai monsoon). 
                Instead, a <b>City × Season Median Imputation</b> strategy was engineered, preserving micro-climatic standard deviations with zero cross-city distribution bleeding.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="section-heading">Operational Boundaries & Scientific Limitations</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:20px 24px; font-size:13.5px; color:#334155; line-height:1.65;">
                Rigorous scientific integrity requires explicit documentation of edge cases and operational constraints:
                <ol style="margin: 8px 0; padding-left: 20px;">
                    <li><b>Geographic Station Coverage (29 Urban Centers):</b> The model is currently trained on 29 major Indian urban centers where multi-year, continuous historical CAAQMS telemetry was officially accessible and verified from CPCB archives. Broad-scale real-time open data across other Indian tier-2 and tier-3 cities could not be extracted at the time of research; extending coverage to additional nationwide corridors is actively planned for upcoming versions.</li>
                    <li><b>Peak Spike Attenuation:</b> Gradient boosted decision trees exhibit regression-to-the-mean during extreme episodic spikes (Actual AQI > 400). Error auditing revealed peak residuals in Gurugram (MAE 20.94) during stubble-burning episodes.</li>
                    <li><b>Zero Fabricated Pollutants Rule:</b> In Public Citizen Mode, the system strictly relies on ambient weather and seasonal spatial coordinates. Unmeasured chemical sensors are populated via station baseline medians rather than synthesized to ensure data governance integrity.</li>
                    <li><b>Sensor Recalibration Drift:</b> CAAQMS optical sensors experience baseline zero-point drifts over 5-year longitudinal periods, requiring periodic recalibration offsets.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_side:
        st.markdown('<div class="section-heading">Project Metadata</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:22px; font-size:13px; line-height:1.65; color:#334155; margin-bottom:18px;">
                <div style="margin-bottom:12px;">
                    <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Principal Researcher</span><br>
                    <b style="font-size:15px; color:#0F172A;">Ritvika</b><br>
                    <span style="color:#2563EB; font-weight:600;">Indian Institute of Technology (BHU)</span>
                </div>
                <div style="margin-bottom:12px;">
                    <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Primary Repository</span><br>
                    <b style="color:#0F172A;">Kritvi0208 / AirIntel</b><br>
                    <span style="font-size:12px; color:#64748B;">National Air Quality Intelligence</span>
                </div>
                <div style="margin-bottom:12px;">
                    <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Technology Stack</span><br>
                    • Python 3.13 • Streamlit • Scikit-Learn<br>
                    • LightGBM • CatBoost • TreeSHAP<br>
                    • MLflow • DVC • Pytest • Docker<br>
                    • Plotly Express • Pandas • NumPy • MkDocs
                </div>
                <div style="margin-bottom:12px;">
                    <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Serving Artifact</span><br>
                    <b style="color:#0F172A;">deployment_pipeline.pkl</b> (3.57 MB)<br>
                    <span style="font-size:11.5px; color:#059669; font-weight:600;">P50: 18.7 ms • P99: 38 ms SLA</span>
                </div>
                <div>
                    <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Licensing</span><br>
                    <b style="color:#0F172A;">MIT Open Source License</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 4. TECHNICAL DOCUMENTATION PORTAL (MKDOCS)
    st.markdown('<div class="section-heading">📚 Comprehensive Technical Documentation (MkDocs)</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 60%, #EFF6FF 100%); border: 1px solid #BFDBFE; border-radius: 14px; padding: 24px 28px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.04);">
            <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px;">
                <div>
                    <div style="font-size: 18px; font-weight: 800; color: #0F172A; display: flex; align-items: center; gap: 8px;">
                        AirIntel MkDocs Technical Documentation Portal
                        <span style="background: #ECFDF5; color: #059669; border: 1px solid #A7F3D0; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">Active on :8000</span>
                    </div>
                    <div style="font-size: 13.5px; color: #475569; margin-top: 4px; line-height: 1.5;">
                        Complete scientific reference documentation generated with <b>Material for MkDocs</b>, featuring mathematical derivations, 14-notebook lifecycle deep-dives, benchmark tables, and API references.
                    </div>
                </div>
                <a href="http://localhost:8000" target="_blank" style="text-decoration: none;">
                    <div style="background: #2563EB; color: #FFFFFF; font-size: 13.5px; font-weight: 700; padding: 9px 18px; border-radius: 8px; display: inline-flex; align-items: center; gap: 6px; box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3);">
                        📖 Open Documentation Portal &rarr;
                    </div>
                </a>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 14px; font-size: 12.5px;">
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px;">
                    <b style="color: #1E293B;">🔬 14-Stage Lifecycle</b><br>
                    <span style="color: #64748B;">Complete provenance of Notebooks 01 to 14.</span>
                </div>
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px;">
                    <b style="color: #1E293B;">📐 Atmospheric Physics</b><br>
                    <span style="color: #64748B;">Boundary layer inversions & scavenging math.</span>
                </div>
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px;">
                    <b style="color: #1E293B;">⚙️ MLOps & Drift Auditing</b><br>
                    <span style="color: #64748B;">DVC, MLflow, Pytest, Docker & PSI equations.</span>
                </div>
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px;">
                    <b style="color: #1E293B;">💻 Python Serving API</b><br>
                    <span style="color: #64748B;">Docstrings & specifications for <code>src/inference.py</code>.</span>
                </div>
            </div>
            <div style="margin-top: 14px; padding-top: 10px; border-top: 1px dashed #CBD5E1; font-size: 12px; color: #64748B; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                <span>Terminal Commands:</span>
                <code style="background: #F1F5F9; color: #0F172A; padding: 2px 6px; border-radius: 4px;">uv run mkdocs serve</code>
                <span>(live reload at <code>http://localhost:8000</code>)</span>
                <span>•</span>
                <code style="background: #F1F5F9; color: #0F172A; padding: 2px 6px; border-radius: 4px;">uv run mkdocs build</code>
                <span>(build static HTML to <code>site/</code>)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
