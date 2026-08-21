import streamlit as st
from pathlib import Path

# 1. Page Configuration (Wide SaaS Layout)
st.set_page_config(
    page_title="AirIntel - National Air Quality Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Light-Blue Tint SaaS Theme Stylesheet
css_path = Path("assets/style.css")
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 3. Load Production Deployment Bundle (Cached)
from components.utils import load_deployment_bundle
pipeline_bundle, bundle_error = load_deployment_bundle("models/deployment/deployment_pipeline.pkl")
valid_cities = pipeline_bundle.get("valid_cities", ["Delhi", "Mumbai", "Bengaluru"]) if pipeline_bundle else ["Delhi", "Mumbai", "Bengaluru"]
feature_medians = pipeline_bundle.get("feature_medians", {}) if pipeline_bundle else {}

# 4. Import Component Modules
from components.sidebar import render_sidebar
from components.overview import render_overview
from components.analytics import render_analytics
from components.prediction import render_prediction_page
from components.maps import render_spatial_analytics
from components.diagnostics import render_explainability, render_system_page, render_about_page
from components.reports import render_reports

# 5. Top Navigation Bar (Robust state management avoiding StreamlitAPIException)
pages = ["Home", "Analytics", "Prediction", "Spatial", "Explainability", "Architecture", "Downloads", "About"]

# Check if a CTA button requested a page transition
if "nav_target" in st.session_state and st.session_state["nav_target"] in pages:
    st.session_state["active_page"] = st.session_state.pop("nav_target")

if "active_page" not in st.session_state or st.session_state["active_page"] not in pages:
    st.session_state["active_page"] = "Home"

selected_page = st.segmented_control(
    "Main Navigation",
    pages,
    default=st.session_state["active_page"],
    key=f"nav_seg_ctrl_{st.session_state.get('nav_counter', 0)}",
    label_visibility="collapsed"
)

if not selected_page:
    selected_page = st.session_state["active_page"]
else:
    st.session_state["active_page"] = selected_page

# 6. Render ONE Dynamic Context-Aware Sidebar
sidebar_output, prediction_mode = render_sidebar(
    selected_page,
    valid_cities,
    feature_medians
)

if bundle_error:
    st.warning(f"Deployment System Notice: {bundle_error}")

# 7. Render Selected Page (Strict single-page execution)
if selected_page == "Home":
    render_overview(pipeline_bundle or {})
elif selected_page == "Analytics":
    render_analytics(pipeline_bundle or {}, sidebar_output)
elif selected_page == "Prediction":
    if pipeline_bundle:
        render_prediction_page(sidebar_output, prediction_mode, pipeline_bundle)
    else:
        st.error("Prediction engine requires the deployment pipeline bundle.")
elif selected_page == "Spatial":
    render_spatial_analytics(pipeline_bundle or {}, sidebar_output)
elif selected_page == "Explainability":
    render_explainability(pipeline_bundle or {}, sidebar_output)
elif selected_page == "Architecture":
    render_system_page(pipeline_bundle or {})
elif selected_page == "Downloads":
    render_reports(sidebar_output, prediction_mode, pipeline_bundle or {})
elif selected_page == "About":
    render_about_page(pipeline_bundle or {})

# 8. Single Clean Application Footer
st.markdown(
    """
    <div class="app-footer">
        AirIntel v2.0 Platform • LightGBM Regressor (R²=0.887) • CatBoost Classifier (89.4%) • TreeSHAP Explainability • Production Deployment Ready
    </div>
    """,
    unsafe_allow_html=True
)
