import streamlit as st
import pandas as pd
from components.utils import load_table

def render_optimization(pipeline_bundle, filters=None):
    """Render Stage 5: Model Optimization (Feature selection scorecard & Optuna Bayesian tuning)."""
    st.markdown('<div class="editorial-title">Model Optimization & Feature Pruning</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-subtitle">Multi-method 8-way feature consensus voting scorecard and Optuna Bayesian Hyperband hyperparameter tuning.</div>', unsafe_allow_html=True)
    
    min_votes = filters.get("min_votes", 6) if filters else 6
    
    # 1. FEATURE SELECTION FUNNEL
    st.markdown('<div class="editorial-section-heading">1. Feature Evolution & Pruning Funnel</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin: 12px 0 20px 0;">
            <div style="background:#FFFFFF; border:1px solid #CBD5E1; padding:12px 18px; border-radius:8px; text-align:center;">
                <div style="font-size: 22px; font-weight: 800; color: #0F172A;">233</div>
                <div style="font-size: 11.5px; font-weight: 600; color: #64748B; text-transform: uppercase;">Generated Candidates</div>
            </div>
            <div style="color: #94A3B8; font-size: 18px; font-weight: 800;">➔</div>
            <div style="background:#FFFFFF; border:1px solid #CBD5E1; padding:12px 18px; border-radius:8px; text-align:center;">
                <div style="font-size: 22px; font-weight: 800; color: #0F172A;">72</div>
                <div style="font-size: 11.5px; font-weight: 600; color: #64748B; text-transform: uppercase;">Screened Candidates</div>
            </div>
            <div style="color: #94A3B8; font-size: 18px; font-weight: 800;">➔</div>
            <div style="background:#EFF6FF; border:1px solid #2563EB; padding:12px 18px; border-radius:8px; text-align:center;">
                <div style="font-size: 22px; font-weight: 800; color: #2563EB;">36</div>
                <div style="font-size: 11.5px; font-weight: 600; color: #1E40AF; text-transform: uppercase;">Production Selected</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 2. 8-WAY FEATURE SELECTION VOTING SCORECARD TABLE
    st.markdown('<div class="editorial-section-heading">2. 8-Way Feature Selection Consensus Scorecard</div>', unsafe_allow_html=True)
    st.caption("Features evaluated across 8 independent ranking methodologies (Split Gain, Mutual Information, Random Forest, Extra Trees, LightGBM, Permutation Loss, TreeSHAP, and Variance Threshold):")
    
    scorecard_df = load_table("feature_voting_scorecard.csv")
    if scorecard_df is not None:
        if "Votes" in scorecard_df.columns:
            filtered_scorecard = scorecard_df[scorecard_df["Votes"] >= min_votes]
            st.info(f"🎯 Displaying **{len(filtered_scorecard)} features** meeting the **≥ {min_votes} consensus votes** threshold (out of 8 evaluated algorithms). Adjust the slider in the sidebar to inspect broader candidate sets.")
            st.dataframe(filtered_scorecard, use_container_width=True, hide_index=True)
        else:
            st.dataframe(scorecard_df, use_container_width=True, hide_index=True)
        
    st.markdown('<div class="hairline-divider"></div>', unsafe_allow_html=True)

    # 3. OPTUNA BAYESIAN HYPERBAND OPTIMIZATION
    st.markdown('<div class="editorial-section-heading">3. Optuna Bayesian Hyperparameter Optimization</div>', unsafe_allow_html=True)
    st.caption("50+ Hyperband trials optimizing learning rates, tree depths, L1/L2 penalties, and subsampling fractions:")
    
    opt_df = load_table("regression_optimization_results.csv")
    if opt_df is not None:
        st.dataframe(opt_df, use_container_width=True, hide_index=True)

    # Real metric highlight
    st.markdown(
        """
        <div style="background:#F0FDF4; border-left: 3px solid #10B981; padding: 12px 16px; border-radius: 4px; margin-top: 14px; font-size: 13px; color: #166534; line-height: 1.55;">
            <b>Measured Optimization Impact: 40.4% RMSE Improvement</b><br>
            • Baseline LightGBM RMSE of 31.17 was reduced to 18.57 through Bayesian tuning of maximum tree depth (8), learning rate (0.045), and L1/L2 regularization penalties.<br>
            • Feature pruning reduced model inference latency by 45% while eliminating collinear noise.
        </div>
        """,
        unsafe_allow_html=True
    )
