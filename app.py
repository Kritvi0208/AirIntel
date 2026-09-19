import streamlit as st
from pathlib import Path

# 1. Page Configuration (Wide Scientific Editorial Layout)
st.set_page_config(
    page_title="AirIntel - National Air Quality Intelligence Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Editorial Scientific Stylesheet
css_path = Path("assets/style.css")
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 3. Load Production Deployment Bundle (Cached)
from components.utils import load_deployment_bundle
pipeline_bundle, bundle_error = load_deployment_bundle("models/deployment/deployment_pipeline.pkl")
valid_cities = pipeline_bundle.get("valid_cities", ["Delhi", "Mumbai", "Bengaluru"]) if pipeline_bundle else ["Delhi", "Mumbai", "Bengaluru"]
feature_medians = pipeline_bundle.get("feature_medians", {}) if pipeline_bundle else {}

# 4. Import 10-Stage Component Modules
from components.sidebar import render_sidebar
from components.overview import render_overview
from components.journey import render_research_journey
from components.eda_stats import render_eda_statistics
from components.modeling import render_modeling
from components.optimization import render_optimization
from components.advanced_analytics import render_advanced_analytics
from components.prediction import render_prediction_page
from components.explainability_spatial import render_explainability_spatial
from components.about import render_about

# 5. Canonical Page Definitions with Sleek Icons
PAGE_CONFIG = {
    "Overview": "🏠 Overview",
    "Research Journey": "🗺️ Journey",
    "EDA & Statistics": "📊 EDA & Stats",
    "Modeling": "📈 Modeling",
    "Optimization": "⚡ Optimization",
    "Advanced Analytics": "🔬 Analytics",
    "Prediction": "🔮 Prediction",
    "Explainability & Spatial": "🌍 Spatial & SHAP",
    "About": "ℹ️ About"
}
CANONICAL_PAGES = list(PAGE_CONFIG.keys())
LABEL_TO_CANONICAL = {v: k for k, v in PAGE_CONFIG.items()}
CANONICAL_TO_LABEL = PAGE_CONFIG

# Check if a CTA button requested a page transition
if "nav_target" in st.session_state and st.session_state["nav_target"] in CANONICAL_PAGES:
    st.session_state["active_page"] = st.session_state.pop("nav_target")

if "active_page" not in st.session_state or st.session_state["active_page"] not in CANONICAL_PAGES:
    st.session_state["active_page"] = "Overview"

# Top Navigation Bar with Sleek Concise Labels
top_labels = list(PAGE_CONFIG.values())
current_label = CANONICAL_TO_LABEL[st.session_state["active_page"]]

selected_label = st.segmented_control(
    "Main Navigation",
    top_labels,
    default=current_label,
    key=f"nav_seg_ctrl_{st.session_state.get('nav_counter', 0)}",
    label_visibility="collapsed"
)

if not selected_label:
    selected_label = current_label

if selected_label in LABEL_TO_CANONICAL:
    st.session_state["active_page"] = LABEL_TO_CANONICAL[selected_label]

selected_page = st.session_state["active_page"]

# 6. Render Dynamic Context-Aware Sidebar tailored to selected page
sidebar_output, prediction_mode = render_sidebar(
    selected_page,
    valid_cities,
    feature_medians
)

if bundle_error:
    st.warning(f"Deployment System Notice: {bundle_error}")

# 7. Render Selected Page (Strict single-page execution across 10-stage story)
if selected_page == "Overview":
    render_overview(pipeline_bundle or {})
elif selected_page == "Research Journey":
    render_research_journey(pipeline_bundle or {})
elif selected_page == "EDA & Statistics":
    render_eda_statistics(pipeline_bundle or {}, sidebar_output)
elif selected_page == "Modeling":
    render_modeling(pipeline_bundle or {}, sidebar_output)
elif selected_page == "Optimization":
    render_optimization(pipeline_bundle or {}, sidebar_output)
elif selected_page == "Advanced Analytics":
    render_advanced_analytics(pipeline_bundle or {}, sidebar_output)
elif selected_page == "Prediction":
    if pipeline_bundle:
        render_prediction_page(sidebar_output, prediction_mode, pipeline_bundle)
    else:
        st.error("Prediction engine requires the deployment pipeline bundle.")
elif selected_page == "Explainability & Spatial":
    render_explainability_spatial(pipeline_bundle or {}, sidebar_output)
elif selected_page == "About":
    render_about(pipeline_bundle or {})

# 8. Single Clean Application Footer
st.markdown(
    """
    <div class="app-footer">
        AirIntel v2.0 Platform • LightGBM Regressor (R²=0.887) • CatBoost Classifier (89.4%) • TreeSHAP Explainability • Production Deployment Ready
    </div>
    """,
    unsafe_allow_html=True
)
