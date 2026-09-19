import streamlit as st
import pandas as pd
import json
import textwrap
from components.utils import load_table

def render_research_journey(pipeline_bundle):
    """Render Stage 2: Research Journey (Interactive 4-Phase Flowchart with Click-to-Inspect Details)."""
    st.markdown('<div class="editorial-title">Research & Engineering Journey</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-subtitle">Interactive 14-stage engineering flowchart detailing exact decisions, transformations, and deliverables across 4 project phases.</div>', unsafe_allow_html=True)
    
    stages = [
        {
            "num": "01",
            "phase": "Phase 1: Ingestion & Validation",
            "short_title": "Data Extraction",
            "title": "Data Ingestion & Multi-Source Extraction",
            "notebook": "01_Data_Extraction.ipynb",
            "scope": "842,160+ hourly records • 29 Indian cities • 6 criteria pollutants • 7 weather variables",
            "what_i_did": [
                "Harmonized heterogeneous CAAQMS station feeds from Central Pollution Control Board (CPCB) with historical OpenWeather archives.",
                "Standardized timestamp schemas across differing station timezones into unified UTC/IST datetime indices.",
                "Merged high-frequency environmental pollutant measurements with synchronous ambient meteorological observations."
            ],
            "outputs": "raw_airintel.parquet, ingestion_report.csv",
            "key_finding": "Established a verified national baseline across 29 major industrial, coastal, and Gangetic basin urban corridors."
        },
        {
            "num": "02",
            "phase": "Phase 1: Ingestion & Validation",
            "short_title": "Data Auditing",
            "title": "Data Auditing & Sensor Anomaly Detection",
            "notebook": "02_Data_Validation.ipynb",
            "scope": "Sensor quality • Missingness rate profiling • Sensor calibration drift • Spatial bounding",
            "what_i_did": [
                "Profiled missing value rates across all 13 chemical and meteorological channels per city and per season.",
                "Detected sensor clipping errors including negative concentration values caused by optical zero-point drift.",
                "Identified flatline zero-variance sequences caused by telemetry dropouts and verified physical coordinate bounding boxes (8.4°N–32.7°N, 69.0°E–92.8°E)."
            ],
            "outputs": "validation_report.csv",
            "key_finding": "Naively dropping missing records would have removed more than 35% of valid time series observations, proving the necessity of an intelligent localized imputation strategy."
        },
        {
            "num": "03",
            "phase": "Phase 1: Ingestion & Validation",
            "short_title": "Variance Cleaning",
            "title": "Variance-Preserving Data Cleaning",
            "notebook": "03_Data_Cleaning.ipynb",
            "scope": "City × Season median imputation • Limit of Detection (LOD) • Physical outlier clipping",
            "what_i_did": [
                "Engineered City × Season Median Imputation: Avoided naive global mean imputation which distorts natural climate variances (e.g. imputing Delhi winter particulate loads into Mumbai monsoon data).",
                "Replaced negative sensor values with analytical Limit of Detection (LOD) lower thresholds rather than zero to maintain mathematical validity for log transforms.",
                "Applied physically bounded clipping on impossible sensor spikes (PM2.5 > 1500 µg/m³) matching EPA and CPCB regulatory limits."
            ],
            "outputs": "clean_airintel.parquet (842,160 clean records, 63 validated columns), data_cleaning_report.csv",
            "key_finding": "Localized imputation preserved micro-climate seasonal standard deviations with zero cross-city distribution bleeding."
        },
        {
            "num": "04",
            "phase": "Phase 2: Atmospheric Discovery",
            "short_title": "Feature Engineering",
            "title": "Atmospheric Domain Feature Engineering",
            "notebook": "04_Feature_Engineering.ipynb",
            "scope": "12 base raw channels ➔ 233 engineered candidate features",
            "what_i_did": [
                "Temporal Cyclical Harmonics: Converted Hour (0–23), Month (1–12), and Weekday into continuous sine/cosine pairs to eliminate boundary cliffs.",
                "Thermodynamics & Aerosols: Formulated Temp × Humidity (hygroscopic secondary aerosol growth) and Dew Point Depression (smog condensation potential).",
                "Atmospheric Physics: Created Planetary Boundary Layer (PBL) Stagnation indicators (Wind < 5 km/h & Temp < 15°C) and Northern India basin flags (Lat > 20°N).",
                "Rolling Dynamics: Computed 12h, 24h, and 7d rolling averages, exponential moving averages (EMAs), and rolling min/max spreads.",
                "Domain Regimes: Tagged the post-monsoon stubble burning harvest window (October 15 – November 30) and festival periods."
            ],
            "outputs": "airintel_features_expanded.parquet (233 features), feature_engineering_report.csv",
            "key_finding": "Atmospheric thermodynamic interactions and cyclical solar harmonics significantly boosted tree-based split gains."
        },
        {
            "num": "05",
            "phase": "Phase 2: Atmospheric Discovery",
            "short_title": "Atmospheric EDA",
            "title": "Exploratory Data Analysis & Discovery",
            "notebook": "05_Advanced_EDA.ipynb",
            "scope": "Diurnal cycle dynamics • Seasonal inversions • Bimodal distributions • Rain scavenging",
            "what_i_did": [
                "Uncovered bimodal diurnal AQI curve peaking during morning rush hour (8–10 AM) and evening inversion trapping (8–11 PM).",
                "Analyzed monsoon scavenging: Verified rainfall wet deposition decreases ambient PM2.5 concentrations by ~75% across coastal and peninsular stations.",
                "Quantified winter boundary layer suppression: Verified a 3.2× pollution surge in landlocked northern basins during December–January.",
                "Evaluated weekend vs weekday dynamics: Uncovered a 14% drop in mobile emission markers (NO2, CO) on Sundays in tier-1 metros."
            ],
            "outputs": "eda_summary.csv, pollutant_summary.csv",
            "key_finding": "Confirmed PM2.5 as India's dominant criterion pollutant, exhibiting a 0.92 Pearson correlation with the composite US AQI."
        },
        {
            "num": "06",
            "phase": "Phase 2: Atmospheric Discovery",
            "short_title": "Statistical Analysis",
            "title": "Statistical Testing & Collinearity Screening",
            "notebook": "06_Statistical_Analysis.ipynb",
            "scope": "Hypothesis testing (ANOVA, Mann-Whitney U, Kruskal-Wallis) • Collinearity screening",
            "what_i_did": [
                "Screened 233 feature candidates through multicollinearity filtering, discarding redundant correlations.",
                "Conducted Kruskal-Wallis non-parametric tests confirming regional air quality divergence between Indo-Gangetic Plains and Peninsular India (H = 186,343.72, p < 10⁻³⁰⁰).",
                "Executed Mann-Whitney U testing on crop burning windows (p < 10⁻¹⁵, Cohen's d = 1.18) and weekend vs weekday traffic emission drops."
            ],
            "outputs": "hypothesis_test_results.csv, statistical_summary.csv",
            "key_finding": "Pruned 161 collinear and redundant features while retaining 72 statistically verified predictors for consensus selection."
        },
        {
            "num": "07",
            "phase": "Phase 3: ML & Optimization",
            "short_title": "AQI Regression",
            "title": "Continuous US AQI Regression Benchmarking",
            "notebook": "07_Regression_Modeling.ipynb",
            "scope": "11 model families • 80/20 Chronological temporal train-test split • 5-fold CV • Zero Data Leakage",
            "what_i_did": [
                "Enforced strict chronological temporal splitting (80% train, 20% test) to prevent temporal data leakage.",
                "Trained 11 regression models: Linear, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, AdaBoost, LightGBM, XGBoost.",
                "Identified Random Forest overfitting: Train R² = 0.981 vs Test R² = 0.852 and heavy 448 MB memory footprint.",
                "Selected LightGBM as primary champion: Validation R² = 0.8449, MAE = 13.32, CPU latency = 18.7ms."
            ],
            "outputs": "regression_results.csv (11 models), regression_leaderboard.csv",
            "key_finding": "Gradient boosted trees significantly outperformed linear baselines (R² 0.845 vs 0.354), successfully capturing non-linear meteorological dispersion."
        },
        {
            "num": "08",
            "phase": "Phase 3: ML & Optimization",
            "short_title": "Severity Classification",
            "title": "Multi-Class Severity Classification",
            "notebook": "08_Classification_Modeling.ipynb",
            "scope": "6 EPA AQI risk categories • 10 model families • Class-balanced weighting",
            "what_i_did": [
                "Trained 10 classification models across 6 discrete EPA health tiers (Good to Hazardous).",
                "Addressed extreme class imbalance (Hazardous class represented <4% of records) using class-weighted sample losses.",
                "Selected CatBoost as primary classifier: Test Accuracy = 89.4%, Weighted F1 = 0.892, macro F1 = 0.756."
            ],
            "outputs": "classification_results.csv (10 models), classification_leaderboard.csv",
            "key_finding": "CatBoost achieved highest balanced recall on hazardous and very unhealthy tiers without sacrificing general precision."
        },
        {
            "num": "09",
            "phase": "Phase 3: ML & Optimization",
            "short_title": "Consensus & Optuna",
            "title": "Feature Consensus Scorecard & Optuna Tuning",
            "notebook": "09_Optimization.ipynb",
            "scope": "8-way consensus voting scorecard • Optuna Hyperband Bayesian tuning (100 trials)",
            "what_i_did": [
                "Constructed an 8-way voting scorecard (Gain, Split, Mutual Info, Random Forest, Extra Trees, Permutation Loss, TreeSHAP, Kendall Tau).",
                "Consensus Pruning: Cut 72 candidates down to the top 36 unanimously agreed production features (saving 50% inference payload).",
                "Executed Optuna Bayesian Optimization over 100 trials with early stopping, tuning LightGBM and CatBoost hyperparameters."
            ],
            "outputs": "feature_voting_scorecard.csv, regression_optimization_results.csv, final_features_used.csv",
            "key_finding": "Optuna Bayesian tuning reduced LightGBM validation RMSE by 40.4% (from 31.17 down to 18.57) and boosted test R² to 0.8874."
        },
        {
            "num": "10",
            "phase": "Phase 4: Spatial & Deploy",
            "short_title": "Spatial Clustering",
            "title": "Unsupervised Clustering & Network Transport",
            "notebook": "10_Spatial_Clustering_Analysis.ipynb",
            "scope": "K-Means & Hierarchical clustering • PCA/t-SNE/UMAP projections • Graph centrality",
            "what_i_did": [
                "Segmented 29 cities into 2 primary archetypes: Cluster 0 (12 clean/coastal cities, mean AQI 69.94) vs Cluster 1 (17 industrial/Gangetic cities, mean AQI 115.77).",
                "Computed real 2D dimensionality reduction coordinates using PCA, t-SNE, and UMAP.",
                "Constructed transboundary pollution transport networks, evaluating Degree, Betweenness, and Closeness centralities."
            ],
            "outputs": "cluster_summary.csv, dim_reduction_projections.csv, city_similarity.csv, network_metrics_summary.csv",
            "key_finding": "Agartala and Raipur act as critical topological betweenness bridges connecting distinct regional pollution regimes."
        },
        {
            "num": "11",
            "phase": "Phase 4: Spatial & Deploy",
            "short_title": "TreeSHAP Explainability",
            "title": "TreeSHAP Explainability & Error Diagnostics",
            "notebook": "11_Model_Explainability.ipynb",
            "scope": "TreeSHAP local & global attribution • Permutation importance • Error hotspot auditing",
            "what_i_did": [
                "Computed TreeSHAP values decomposing continuous predictions into additive feature attribution contributions.",
                "Audited residual failure hotspots: Identified Gurugram (MAE 20.94) and Delhi (MAE 19.55) as peak dispersion variance regions.",
                "Audited worst test errors: Discovered tree shrinkage under-predicts extreme off-scale spike episodes (Actual AQI > 450)."
            ],
            "outputs": "permutation_importance.csv, city_error_summary.csv, season_error_summary.csv, worst_predictions.csv",
            "key_finding": "Established Northern_India, Temp_2m_C, and Monsoon season as dominant drivers shifting predictions relative to the base value."
        },
        {
            "num": "12",
            "phase": "Phase 4: Spatial & Deploy",
            "short_title": "Production Packaging",
            "title": "Pipeline Serialization & Production Packaging",
            "notebook": "12_Deployment_Packaging.ipynb",
            "scope": "deployment_pipeline.pkl (3.57 MB) • Measured SLAs (P50: 22ms, P99: 38ms) • Dual REST API contracts",
            "what_i_did": [
                "Encapsulated preprocessing, StandardScaler, OneHotEncoder, and trained models into a single deployable asset (deployment_pipeline.pkl).",
                "Benchmarked real CPU inference latency: P50 latency of 22ms, P99 latency of 38ms.",
                "Engineered Dual-Contract Architecture: Scientific Mode (full pollutant array for chemical diagnostics) vs Public Citizen Mode (Option A: weather-driven interface awaiting forecast model, zero fabricated pollutants)."
            ],
            "outputs": "deployment_pipeline.pkl, prediction_schema.json, sample_request.json, deployment_summary.csv",
            "key_finding": "Unified pipeline serialization ensures zero feature schema drift between research training and live production serving."
        },
        {
            "num": "13",
            "phase": "Phase 4: Spatial & Deploy",
            "short_title": "SaaS Platform",
            "title": "Production SaaS Web Platform Architecture",
            "notebook": "13_Streamlit_Dashboard.ipynb",
            "scope": "Streamlit app.py • 10-Stage scientific story • High-performance visual UI",
            "what_i_did": [
                "Architected a single-page conditional execution routing engine in app.py, executing only the active view per user interaction.",
                "Enforced a single sidebar render pass per rerun, eliminating duplicate computations.",
                "Built the 10-stage scientific editorial narrative interface with interactive Plotly OpenStreetMap visualizers, live TreeSHAP waterfall charts, and diagnostic data exporters."
            ],
            "outputs": "Live AirIntel SaaS Web Platform (http://localhost:8502), Git repository sync",
            "key_finding": "Translated complex multi-notebook atmospheric ML pipelines into a production-grade, transparent intelligence product."
        },
        {
            "num": "14",
            "phase": "Phase 4: Spatial & Deploy",
            "short_title": "MLOps & Monitoring",
            "title": "MLOps, Reproducibility & Production Drift Monitoring",
            "notebook": "14_MLOps_Reproducibility_Monitoring.ipynb",
            "scope": "MLflow tracking • DVC data versioning • Pytest suite (5 core tests) • Docker containerization • PSI & KS drift monitoring",
            "what_i_did": [
                "Experiment Tracking: Codified LightGBM continuous regression (R²=0.8245, MAE=12.46) and CatBoost classification benchmarks in local SQLite-backed MLflow.",
                "Automated Verification: Engineered a 5-point Pytest suite (data integrity, deployment bundle, sub-50ms SLA, calibrated probabilities, and anti-leakage audit).",
                "Continuous Integration: Built GitHub Actions workflows (.github/workflows/tests.yml & deploy.yml) and production Dockerfile (python:3.12-slim).",
                "Telemetry & Drift: Implemented streaming JSONL inference telemetry (logs/predictions.jsonl) and real-time statistical drift auditing using Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS) tests."
            ],
            "outputs": "14_MLOps_Reproducibility_Monitoring.ipynb, mlflow.db, dvc.yaml, tests/test_pipeline.py, Dockerfile, logs/predictions.jsonl",
            "key_finding": "Established end-to-end production governance, sub-50ms inference verification, and automated drift alerting without modifying trained models or interrupting live serving."
        }
    ]
    
    if "journey_stage_idx" not in st.session_state:
        st.session_state["journey_stage_idx"] = 0
        
    current_idx = st.session_state["journey_stage_idx"]
    
    # -------------------------------------------------------------
    # 4 EXPANDABLE VERTICAL TIMELINE PHASES
    # -------------------------------------------------------------
    st.markdown('<div class="editorial-section-heading">Research & Engineering Pipeline Progression</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:13.5px; color:#475569; margin-bottom: 22px;">
            The project follows a 14-stage scientific progression divided across 4 distinct phases. 
            Select any phase container below to expand its vertical engineering timeline.
        </div>
        """,
        unsafe_allow_html=True
    )

    phases = [
        {
            "id": "phase1",
            "phase_tag": "Phase 1",
            "title": "Phase 1: Ingestion & Sensor Quality Auditing (Stages 01–03)",
            "badge_color": "#2563EB",
            "summary_preview": "842,160+ Records • 29 Urban Centers • Sensor Drift Profile • City × Season Median Imputation",
            "subtitle": "Raw telemetry harmonization, sensor anomaly profiling, and variance-preserving median imputation across 29 cities.",
            "stages": stages[0:3],
            "default_expanded": True
        },
        {
            "id": "phase2",
            "phase_tag": "Phase 2",
            "title": "Phase 2: Atmospheric Physics & Feature Discovery (Stages 04–06)",
            "badge_color": "#0D9488",
            "summary_preview": "233 Feature Candidates • Boundary Layer Inversion • Diurnal Harmonics • Collinearity Screening",
            "subtitle": "233 candidate features engineered from thermodynamics, solar harmonics, and collinearity screening.",
            "stages": stages[3:6],
            "default_expanded": False
        },
        {
            "id": "phase3",
            "phase_tag": "Phase 3",
            "title": "Phase 3: Machine Learning & Optuna Optimization (Stages 07–09)",
            "badge_color": "#7C3AED",
            "summary_preview": "11 Regression Models • 10 Classifiers • 8-Way Consensus Scorecard • 40.4% RMSE Reduction",
            "subtitle": "11 regression models, 10 classification models, 8-way voting consensus scorecard, and Optuna Bayesian tuning.",
            "stages": stages[6:9],
            "default_expanded": False
        },
        {
            "id": "phase4",
            "phase_tag": "Phase 4",
            "title": "Phase 4: Spatial Analytics, Deployment & MLOps (Stages 10–14)",
            "badge_color": "#4338CA",
            "summary_preview": "Spatial Clustering • TreeSHAP • 3.57 MB Bundle • MLflow Tracking • Pytest Suite • PSI Drift",
            "subtitle": "Macro-clustering, TreeSHAP explainability, 3.57 MB serialized bundle, MLflow tracking, automated testing, and statistical drift monitoring.",
            "stages": stages[9:14],
            "default_expanded": False
        }
    ]

    for p in phases:
        # Phase Indicator Strip that keeps expander cards engaging when closed
        st.markdown(
            textwrap.dedent(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; background: #F8FAFC; border: 1px solid #E2E8F0; border-bottom: none; border-radius: 10px 10px 0 0; padding: 7px 14px; margin-bottom: 0px;">
                <span style="background: {p['badge_color']}; color: #FFFFFF; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 6px; letter-spacing: 0.03em;">{p['phase_tag']}</span>
                <span style="font-size: 11.5px; color: #64748B; font-weight: 500;">{p['summary_preview']}</span>
            </div>
            """),
            unsafe_allow_html=True
        )
        
        with st.expander(p['title'], expanded=p['default_expanded']):
            st.markdown(f"<div style='font-size:13px; color:#64748B; margin-bottom:18px;'>{p['subtitle']}</div>", unsafe_allow_html=True)
            
            for s in p["stages"]:
                actions_html = "".join([f"<li style='margin-bottom:6px;'>{act}</li>" for act in s["what_i_did"]])
                
                timeline_card_html = textwrap.dedent(f"""
                <div class="timeline-item">
                    <div class="timeline-dot">{s['num']}</div>
                    <div class="timeline-card">
                        <div class="timeline-title-row">
                            <div class="timeline-title">{s['title']}</div>
                            <span class="timeline-nb-badge">{s['notebook']}</span>
                        </div>
                        <div style="font-size: 12px; font-weight: 600; color: #64748B; margin-bottom: 10px;">
                            <b>Scope & Scale:</b> {s['scope']}
                        </div>
                        <div style="font-size: 13px; color: #334155; margin-bottom: 10px;">
                            <b>Technical Actions Performed:</b>
                            <ul style="margin-top: 6px; padding-left: 18px; line-height: 1.55;">
                                {actions_html}
                            </ul>
                        </div>
                        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 12px; margin-bottom: 10px;">
                            <span style="font-weight: 700; color: #0F172A;">Deliverables:</span>
                            <span style="background: #F1F5F9; color: #334155; font-family: monospace; padding: 2px 8px; border-radius: 4px; border: 1px solid #E2E8F0;">{s['outputs']}</span>
                        </div>
                        <div class="timeline-callout">
                            <b>Key Scientific Finding / Breakthrough:</b><br>
                            {s['key_finding']}
                        </div>
                    </div>
                </div>
                """)
                st.markdown(timeline_card_html, unsafe_allow_html=True)
            
            # If Phase 4, cleanly embed the Deployment Specifications & REST Contract
            if p["id"] == "phase4":
                st.markdown('<div class="editorial-section-heading">Production Deployment Specifications & REST Contract</div>', unsafe_allow_html=True)
                st.caption("Microservice operational parameters verified in 12_Deployment_Packaging.ipynb:")
                
                d1, d2, d3, d4 = st.columns(4)
                with d1:
                    st.markdown(
                        """
                        <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:14px; border-radius:8px; text-align:center;">
                            <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Bundle Size</div>
                            <div style="font-size:22px; font-weight:800; color:#0F172A; margin:3px 0;">3.57 MB</div>
                            <div style="font-size:11.5px; color:#475569;">deployment_pipeline.pkl</div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                with d2:
                    st.markdown(
                        """
                        <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:14px; border-radius:8px; text-align:center;">
                            <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">P50 Latency</div>
                            <div style="font-size:22px; font-weight:800; color:#10B981; margin:3px 0;">18.7 ms</div>
                            <div style="font-size:11.5px; color:#475569;">Median CPU Serving</div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                with d3:
                    st.markdown(
                        """
                        <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:14px; border-radius:8px; text-align:center;">
                            <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">P99 Latency SLA</div>
                            <div style="font-size:22px; font-weight:800; color:#2563EB; margin:3px 0;">38.0 ms</div>
                            <div style="font-size:11.5px; color:#475569;">Production SLA Target</div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                with d4:
                    st.markdown(
                        """
                        <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:14px; border-radius:8px; text-align:center;">
                            <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">RAM Footprint</div>
                            <div style="font-size:22px; font-weight:800; color:#7C3AED; margin:3px 0;">~185 MB</div>
                            <div style="font-size:11.5px; color:#475569;">Container Runtime</div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                    
                with st.expander("OpenAPI REST Schema Contracts (POST /predict)", expanded=False):
                    req_col, res_col = st.columns(2)
                    with req_col:
                        st.caption("Standard Input Payload:")
                        sample_req = {
                            "City": "Delhi",
                            "Temp_2m_C": 25.0,
                            "Humidity_Percent": 55.0,
                            "Wind_Speed_10m_kmh": 10.0,
                            "Rain_mm": 0.0,
                            "Month": 11,
                            "Hour": 18
                        }
                        st.code(json.dumps(sample_req, indent=2), language="json")
                    with res_col:
                        st.caption("Production Response Object:")
                        sample_res = {
                            "Prediction_Engine": "AirIntel v2.0",
                            "Predicted_US_AQI": 187.4,
                            "EPA_Severity_Tier": "Unhealthy",
                            "Confidence": 0.892,
                            "Inference_Latency_ms": 18.7
                        }
                        st.code(json.dumps(sample_res, indent=2), language="json")
            
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Deliverables Roadmap Summary Table at Bottom
    with st.expander("Complete 14-Stage Deliverables Roadmap at a Glance", expanded=False):
        roadmap_data = [
            {"Stage": stg["num"], "Notebook": stg["notebook"], "Phase": stg["phase"].split(":")[1].strip(), "Deliverable Title": stg["title"], "Key Output Artifact": stg["outputs"].split(",")[0]}
            for stg in stages
        ]
        st.dataframe(pd.DataFrame(roadmap_data), use_container_width=True, hide_index=True)
