import streamlit as st
import pandas as pd
from components.utils import load_table

def render_modeling(pipeline_bundle, filters=None):
    """Render Stage 4: Modeling (All 11+ models tested, leaderboards, and champion highlights)."""
    st.markdown('<div class="editorial-title">Machine Learning Modeling & Leaderboards</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-subtitle">Comprehensive benchmarking across 11 continuous regression models and 10 severity classification models under a verified Zero Data Leakage Protocol.</div>', unsafe_allow_html=True)
    
    family_choice = filters.get("family", "All 11 Models Tested") if filters else "All 11 Models Tested"
    
    tab_reg, tab_cls = st.tabs([
        "📈 Continuous AQI Regression (11 Models Tested)",
        "🏷️ Severity Classification (10 Models Tested)"
    ])
    
    # =========================================================
    # TAB 1: CONTINUOUS REGRESSION BENCHMARKS
    # =========================================================
    with tab_reg:
        # Champion Model Card (Native Streamlit Container)
        with st.container(border=True):
            col_t1, col_t2 = st.columns([3, 1])
            with col_t1:
                st.markdown("### 🏆 Production Champion: LightGBM Regressor (Optuna Tuned)")
                st.caption("Selected as Champion Asset for Continuous US AQI Microservice")
            with col_t2:
                st.markdown("🎯 **Status: Production Deployed**")
                
            st.divider()
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Test R² Score", "0.8874", "Val R²: 0.8449")
            m2.metric("Test MAE", "14.32", "Val MAE: 13.32")
            m3.metric("Test RMSE", "22.15", "Val RMSE: 18.57")
            m4.metric("CPU Latency", "18.7 ms", "SLA: 38 ms P99")
            m5.metric("Bundle Size", "3.57 MB", "RAM: ~185 MB")
            
            st.divider()
            
            st.info(
                "**Why LightGBM Won:** LightGBM achieved the highest generalized test accuracy (0.8874 R², 14.32 MAE) "
                "without memorizing training noise. In contrast, Random Forest was rejected due to severe overfitting "
                "(Train R² 0.981 vs Test R² 0.852) and an unacceptable 448 MB disk footprint that violates container latency SLAs."
            )

        # Sub-View Selector for Regression
        reg_view = st.radio(
            "Regression Benchmarks View",
            ["🏆 Top Tuned Leaderboard (4 Models)", "📊 Full Model Evaluation (All 11 Models Tested)", "⚠️ Overfitting Audit: Random Forest vs LightGBM"],
            horizontal=True
        )
        
        if "Top Tuned Leaderboard" in reg_view:
            reg_leader = load_table("regression_leaderboard.csv")
            if reg_leader is not None:
                st.caption("Top gradient-boosted models benchmarked with 5-fold cross-validation and test set evaluation:")
                st.dataframe(reg_leader, use_container_width=True, hide_index=True)
                
        elif "Full Model Evaluation" in reg_view:
            reg_all = load_table("regression_results.csv")
            if reg_all is not None:
                if "Gradient Boosted" in family_choice:
                    reg_all = reg_all[reg_all["Model"].str.contains("LightGBM|CatBoost|XGBoost|Gradient|HistGradient", case=False, na=False)]
                    st.caption("Showing **Gradient Boosted Trees** (filtered via sidebar):")
                elif "Ensemble Bagging" in family_choice:
                    reg_all = reg_all[reg_all["Model"].str.contains("Forest|Tree", case=False, na=False)]
                    st.caption("Showing **Ensemble Bagging Models** (filtered via sidebar):")
                elif "Linear" in family_choice:
                    reg_all = reg_all[reg_all["Model"].str.contains("Linear|Ridge|Lasso|ElasticNet", case=False, na=False)]
                    st.caption("Showing **Linear & Regularized Baselines** (filtered via sidebar):")
                else:
                    st.caption("Complete 11-model evaluation spectrum evaluated in `07_Regression_Modeling.ipynb` across linear, ensemble, and boosting architectures:")
                st.dataframe(reg_all, use_container_width=True, hide_index=True)
                
        elif "Overfitting Audit" in reg_view:
            with st.container(border=True):
                st.markdown("#### ⚠️ Technical Justification: Why Random Forest Was Rejected Despite High Train R²")
                st.markdown(
                    "- **Severe Overfitting Delta**: Random Forest achieved Train R² = 0.981 but dropped to Test R² = 0.852 (a 0.129 drop in R²), indicating heavy memorization of leaf noise rather than generalizable atmospheric dynamics.\n"
                    "- **Memory Footprint Bloat**: The unpruned Random Forest model weighed **448 MB** on disk, threatening out-of-memory crashes on cloud microservice containers. In contrast, LightGBM is only **3.57 MB**.\n"
                    "- **Inference Latency**: Random Forest required 142 ms per prediction batch (failing the 50 ms SLA), whereas LightGBM scored in 18.7 ms.\n"
                    "- **Linear Models Failure**: Linear, Ridge, and Lasso regressions plateaued at R² ≈ 0.335 and MAE ≈ 27.3, failing completely to capture non-linear boundary layer stagnation and chemical interactions."
                )
            
        st.divider()
        st.caption("🔒 **Zero Data Leakage Protocol:** Chronological train-test split (80/20) with scalers and encoders fitted exclusively on training splits.")

    # =========================================================
    # TAB 2: MULTI-CLASS CLASSIFICATION BENCHMARKS
    # =========================================================
    with tab_cls:
        # Champion Model Card (Native Streamlit Container)
        with st.container(border=True):
            col_c1, col_c2 = st.columns([3, 1])
            with col_c1:
                st.markdown("### 🏆 Production Champion: CatBoost Multi-Class Classifier")
                st.caption("Selected as Champion Asset for 6-Class EPA Severity Tier Microservice")
            with col_c2:
                st.markdown("🎯 **Status: Production Deployed**")
                
            st.divider()
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Test Accuracy", "89.4%", "Val Acc: 75.6%")
            c2.metric("Weighted F1", "0.892", "Macro F1: 0.679")
            c3.metric("Precision", "0.748", "Balanced")
            c4.metric("Recall", "0.643", "Hazardous: 0.81")
            c5.metric("CPU Latency", "45.7 ms", "SLA: 50 ms")
            
            st.divider()
            
            st.info(
                "**Why CatBoost Won:** CatBoost handles categorical geographic features and class-imbalanced health tiers natively "
                "using symmetric trees, achieving the highest true-positive recall on critical 'Very Unhealthy' and 'Hazardous' episodes."
            )

        # Sub-View Selector for Classification
        cls_view = st.radio(
            "Classification Benchmarks View",
            ["🏆 Top Tuned Leaderboard (4 Models)", "📊 Full Model Evaluation (All 10 Models Tested)", "📋 EPA 6-Tier Risk Definitions"],
            horizontal=True
        )
        
        if "Top Tuned Leaderboard" in cls_view:
            cls_leader = load_table("classification_leaderboard.csv")
            if cls_leader is not None:
                st.caption("Top gradient-boosted multi-class models benchmarked with class-balanced weighting:")
                st.dataframe(cls_leader, use_container_width=True, hide_index=True)
                
        elif "Full Model Evaluation" in cls_view:
            cls_all = load_table("classification_results.csv")
            if cls_all is not None:
                if "Gradient Boosted" in family_choice:
                    cls_all = cls_all[cls_all["Model"].str.contains("LightGBM|CatBoost|XGBoost|Gradient|HistGradient", case=False, na=False)]
                    st.caption("Showing **Gradient Boosted Classifiers** (filtered via sidebar):")
                elif "Ensemble Bagging" in family_choice:
                    cls_all = cls_all[cls_all["Model"].str.contains("Forest|Tree|Extra", case=False, na=False)]
                    st.caption("Showing **Tree Ensemble Classifiers** (filtered via sidebar):")
                elif "Linear" in family_choice:
                    cls_all = cls_all[cls_all["Model"].str.contains("Logistic|Linear|Ridge|SGD|Naive", case=False, na=False)]
                    st.caption("Showing **Linear & Probabilistic Classifiers** (filtered via sidebar):")
                else:
                    st.caption("Complete 10-model evaluation spectrum evaluated in `08_Classification_Modeling.ipynb` across trees, bayes, knn, and boosting architectures:")
                st.dataframe(cls_all, use_container_width=True, hide_index=True)
                
        elif "EPA 6-Tier Risk Definitions" in cls_view:
            epa_tiers = pd.DataFrame([
                {"Category": "Good", "AQI Range": "0 – 50", "Health Impact": "Air quality is considered satisfactory, and air pollution poses little or no risk.", "Tier Color": "🟢 Green"},
                {"Category": "Moderate", "AQI Range": "51 – 100", "Health Impact": "Air quality is acceptable; however, sensitive individuals may experience mild respiratory symptoms.", "Tier Color": "🟡 Yellow"},
                {"Category": "Unhealthy for Sensitive Groups", "AQI Range": "101 – 150", "Health Impact": "Members of sensitive groups may experience health effects. The general public is not likely to be affected.", "Tier Color": "🟠 Orange"},
                {"Category": "Unhealthy", "AQI Range": "151 – 200", "Health Impact": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.", "Tier Color": "🔴 Red"},
                {"Category": "Very Unhealthy", "AQI Range": "201 – 300", "Health Impact": "Health alert: The risk of health effects is increased for everyone.", "Tier Color": "🟣 Purple"},
                {"Category": "Hazardous", "AQI Range": "301 – 500", "Health Impact": "Health warning of emergency conditions: The entire population is more likely to be affected.", "Tier Color": "🟤 Maroon"}
            ])
            st.dataframe(epa_tiers, use_container_width=True, hide_index=True)
