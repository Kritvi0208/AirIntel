import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from components.utils import load_table

def render_advanced_analytics(pipeline_bundle, filters=None):
    """Render Stage 6: Advanced Analytics (Selected interesting results from Notebook 10)."""
    st.markdown('<div class="editorial-title">Advanced Spatial Analytics & Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-subtitle">Urban clustering archetypes, PCA latent projections, inter-city similarity, and pollution transport networks across 29 urban centers.</div>', unsafe_allow_html=True)
    
    cluster_filter = filters.get("cluster", "Both Clusters (29 Cities)") if filters else "Both Clusters (29 Cities)"
    
    # 1. CITY CLUSTERING ARCHETYPES
    st.markdown('<div class="editorial-section-heading">1. Urban Pollution Cluster Archetypes</div>', unsafe_allow_html=True)
    st.caption("K-Means and Hierarchical Clustering segmenting 29 national corridors into two distinct vulnerability regimes:")
    
    cluster_df = load_table("cluster_summary.csv")
    if cluster_df is not None:
        if "Cluster 0" in cluster_filter:
            st.info("📍 Sidebar filter active: Displaying **Cluster 0: Clean Air / Humid Coastal & Tropical Regimes** (12 Cities).")
            st.dataframe(cluster_df[cluster_df["Cluster"] == 0], use_container_width=True, hide_index=True)
        elif "Cluster 1" in cluster_filter:
            st.info("📍 Sidebar filter active: Displaying **Cluster 1: Moderate to Severe / Hot Dry Inland Regimes** (17 Cities).")
            st.dataframe(cluster_df[cluster_df["Cluster"] == 1], use_container_width=True, hide_index=True)
        else:
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)
        
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:10px; padding:18px; font-size:13px; color:#065F46; line-height:1.6;">
                <b style="font-size:14.5px;">Cluster 0: Clean Air / Humid Coastal & Tropical (12 Cities)</b><br>
                <span style="color:#047857; font-size:12.5px;"><i>Bengaluru, Aizawl, Dehradun, Gangtok, Guwahati, Imphal, Itanagar, Kohima, Panaji, Shillong, Shimla, Thiruvananthapuram</i></span><br>
                <div style="margin-top:8px;">
                    • <b>Mean AQI</b>: 69.94 • <b>Mean Temp</b>: 20.59°C • <b>Mean Rainfall</b>: 0.25 mm/h
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div style="background:#FFF1F2; border:1px solid #FECDD3; border-radius:10px; padding:18px; font-size:13px; color:#9F1239; line-height:1.6;">
                <b style="font-size:14.5px;">Cluster 1: Moderate to Severe / Hot Dry Inland (17 Cities)</b><br>
                <span style="color:#BE123C; font-size:12.5px;"><i>Delhi, Gurugram, Lucknow, Patna, Jaipur, Kolkata, Mumbai, Ahmedabad, Bhopal, Bhubaneswar, Chandigarh, Chennai, Hyderabad, Raipur, Ranchi, Visakhapatnam, Agartala</i></span><br>
                <div style="margin-top:8px;">
                    • <b>Mean AQI</b>: 115.77 • <b>Mean Temp</b>: 25.77°C • <b>Mean Rainfall</b>: 0.16 mm/h
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="hairline-divider"></div>', unsafe_allow_html=True)

    # 2. DIMENSIONALITY REDUCTION (PCA Latent Projection)
    st.markdown('<div class="editorial-section-heading">2. PCA High-Dimensional Latent Projections</div>', unsafe_allow_html=True)
    st.caption("Principal Component Analysis projecting 36 atmospheric features into 2D orthogonal variance dimensions:")
    
    proj_df = load_table("dim_reduction_projections.csv")
    if proj_df is not None and "PCA1" in proj_df.columns and "PCA2" in proj_df.columns:
        # Sample 2,500 points for instant 60fps rendering without browser freezing
        df_pca = proj_df.sample(min(len(proj_df), 2500), random_state=42)
        fig_proj = px.scatter(
            df_pca, x="PCA1", y="PCA2", color="US_AQI", hover_name="City",
            color_continuous_scale="Reds", title="PCA Latent Projection of Urban Air Profiles (2,500 Sampled Observations)"
        )
        fig_proj.update_layout(height=400, margin=dict(l=10, r=10, t=35, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_proj, use_container_width=True)

    st.markdown('<div class="hairline-divider"></div>', unsafe_allow_html=True)

    # 3. INTER-CITY SIMILARITY & NETWORK CO-OCCURRENCE
    st.markdown('<div class="editorial-section-heading">3. Pairwise City Similarity & Transport Network</div>', unsafe_allow_html=True)
    sim_df = load_table("city_similarity.csv")
    if sim_df is not None:
        with st.expander("View Pairwise City Similarity Matrix Heatmap", expanded=False):
            numeric_sim = sim_df.select_dtypes(include=[np.number])
            if not numeric_sim.empty:
                fig_sim = px.imshow(numeric_sim.head(15), color_continuous_scale="Viridis", title="Cosine Similarity Across Urban Pollution Signatures")
                fig_sim.update_layout(height=350, margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_sim, use_container_width=True)
                
    net_df = load_table("network_metrics_summary.csv")
    if net_df is not None:
        st.caption("Network Graph Topologies (pollution_network.graphml):")
        st.dataframe(net_df, use_container_width=True, hide_index=True)
