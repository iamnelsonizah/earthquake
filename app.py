import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from database import get_data
from sklearn.cluster import KMeans
import numpy as np

# Set page config
st.set_page_config(page_title="Central Asia Earthquake Dashboard", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    body { font-family: 'Arial', sans-serif; }
    .stApp { background-color: #f9fafb; }
    .header { background-color: #1f2937; color: white; padding: 1rem; border-radius: 0.5rem; }
    .card { background-color: white; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    .metric { font-size: 1.5rem; font-weight: bold; color: #111827; }
    .subheader { color: #374151; font-weight: 600; font-size: 1.25rem; }
    .footer { background-color: #e5e7eb; padding: 1rem; border-radius: 0.5rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Load data
df = get_data()
if df.empty:
    st.error("No data loaded. Please check database initialization.")
    st.stop()

# Sidebar filters
st.sidebar.markdown("<div class='card'><h2 class='subheader'>Filters</h2>", unsafe_allow_html=True)
countries = st.sidebar.multiselect("Country", options=sorted(df['Country'].unique()), default=df['Country'].unique())
min_mag, max_mag = st.sidebar.slider("Magnitude Range", float(df['Magnitude'].min()), float(df['Magnitude'].max()), (float(df['Magnitude'].min()), float(df['Magnitude'].max())))
min_depth, max_depth = st.sidebar.slider("Depth Range (km)", float(df['Depth'].min()), float(df['Depth'].max()), (float(df['Depth'].min()), float(df['Depth'].max())))
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Filter data
filtered_df = df[
    (df['Country'].isin(countries)) &
    (df['Magnitude'].between(min_mag, max_mag)) &
    (df['Depth'].between(min_depth, max_depth))
]

# Title and dataset info
st.markdown("<div class='header'><h1>Central Asia Earthquake Dashboard</h1></div>", unsafe_allow_html=True)
st.markdown("""
<div class='card'>
<p><strong>Dataset:</strong> Central Asia Earthquake 2019-2023, sourced from <a href='https://www.kaggle.com/datasets/sitbayevalibek/central-asian-earthquake-dataset?resource=download' target='_blank'>Kaggle</a>.</p>
<p>Explore earthquake patterns with interactive visualizations and predictions.</p>
</div>
""", unsafe_allow_html=True)

# Tabs for organization
tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Analysis", "Predictions"])

with tab1:
    # Key Metrics
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<p class='metric'>Average Magnitude<br>{filtered_df['Magnitude'].mean():.2f}</p>", unsafe_allow_html=True)
    col2.markdown(f"<p class='metric'>Highest Magnitude<br>{filtered_df['Magnitude'].max():.2f}</p>", unsafe_allow_html=True)
    col3.markdown(f"<p class='metric'>Total Earthquakes<br>{len(filtered_df)}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Earthquake Locations (Interactive Map)
    st.markdown("<div class='card'><h2 class='subheader'>Earthquake Locations</h2>", unsafe_allow_html=True)
    fig_map = px.scatter_geo(
        filtered_df,
        lat='Latitude',
        lon='Longitude',
        size='Magnitude',
        color='Magnitude',
        hover_data=['Country', 'Depth', 'Magnitude'],
        color_continuous_scale='Viridis',
        title="Epicenters (Size and Color by Magnitude)",
        opacity=0.7
    )
    fig_map.update_layout(geo=dict(showcountries=True, countrycolor="gray"), margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Density Heatmap
    st.markdown("<div class='card'><h2 class='subheader'>Earthquake Density Heatmap</h2>", unsafe_allow_html=True)
    fig_heatmap = px.density_mapbox(
        filtered_df,
        lat='Latitude',
        lon='Longitude',
        z='Magnitude',
        radius=10,
        center=dict(lat=filtered_df['Latitude'].mean(), lon=filtered_df['Longitude'].mean()),
        zoom=4,
        mapbox_style="open-street-map",
        title="Earthquake Density (Weighted by Magnitude)",
        opacity=0.6
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    # Magnitude and Depth Distribution
    st.markdown("<div class='card'><h2 class='subheader'>Magnitude and Depth Distribution</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_hist_mag = px.histogram(filtered_df, x='Magnitude', nbins=30, title="Magnitude Distribution")
        st.plotly_chart(fig_hist_mag, use_container_width=True)
    with col2:
        fig_box_depth = px.box(filtered_df, y='Depth', title="Depth Distribution (km)")
        st.plotly_chart(fig_box_depth, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # By Magnitude Category (Bar Chart)
    st.markdown("<div class='card'><h2 class='subheader'>Earthquakes by Magnitude Category</h2>", unsafe_allow_html=True)
    bins = [0, 3, 5, 10]
    labels = ['Minor (<3)', 'Moderate (3-5)', 'Major (>5)']
    filtered_df['Mag_Category'] = pd.cut(filtered_df['Magnitude'], bins=bins, labels=labels, include_lowest=True)
    category_counts = filtered_df.groupby(['Mag_Category', 'Country'], as_index=False).size()
    fig_bar = px.bar(category_counts, x='Mag_Category', y='size', color='Country', barmode='group', title="Count by Magnitude Category")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Top Earthquake-Prone Areas
    st.markdown("<div class='card'><h2 class='subheader'>Top Earthquake-Prone Areas</h2>", unsafe_allow_html=True)
    top_areas = filtered_df.groupby('Country').agg({'Magnitude': 'mean', 'Country': 'count'}).rename(columns={'Country': 'Count', 'Magnitude': 'Avg_Magnitude'}).reset_index()
    top_areas = top_areas.sort_values('Count', ascending=False).head(5)
    fig_top = px.bar(top_areas, x='Country', y='Count', color='Avg_Magnitude', title="Top 5 Countries by Earthquake Count")
    st.plotly_chart(fig_top, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    # Prediction: High-Risk Zones
    st.markdown("<div class='card'><h2 class='subheader'>Predicted High-Risk Zones</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p><strong>Note:</strong> This prediction uses K-Means clustering to identify high-risk zones based on historical earthquake locations and magnitudes. Areas with frequent or high-magnitude earthquakes are marked as higher risk. Predictions are spatial only due to absence of time data.</p>
    """, unsafe_allow_html=True)
    if len(filtered_df) > 10:
        # Prepare data for clustering
        X = filtered_df[['Longitude', 'Latitude', 'Magnitude']].dropna()
        kmeans = KMeans(n_clusters=5, random_state=42).fit(X)
        filtered_df['Risk_Zone'] = kmeans.labels_
        # Calculate risk score
        risk_scores = filtered_df.groupby('Risk_Zone').agg({'Magnitude': 'mean', 'Longitude': 'mean', 'Latitude': 'mean', 'Country': 'first'}).reset_index()
        risk_scores['Risk_Score'] = risk_scores['Magnitude'] / risk_scores['Magnitude'].max()
        # Plot risk zones with annotations
        fig_risk = px.scatter_geo(
            risk_scores,
            lat='Latitude',
            lon='Longitude',
            size='Risk_Score',
            color='Risk_Score',
            hover_data=['Country', 'Risk_Score', 'Magnitude'],
            color_continuous_scale='Reds',
            title="Predicted High-Risk Zones (Color and Size by Risk Score)",
            opacity=0.8,
            text=risk_scores['Risk_Zone'].astype(str)
        )
        fig_risk.update_traces(textposition='top center', textfont=dict(size=12, color='black'))
        fig_risk.update_layout(geo=dict(showcountries=True, countrycolor="gray"), showlegend=False)
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.warning("Insufficient data for reliable predictions.")
    st.markdown("</div>", unsafe_allow_html=True)

# 3D Visualization
st.markdown("<div class='card'><h2 class='subheader'>3D Earthquake Visualization</h2>", unsafe_allow_html=True)
fig_3d = go.Figure(data=[go.Scatter3d(
    x=filtered_df['Longitude'],
    y=filtered_df['Latitude'],
    z=filtered_df['Depth'],
    mode='markers',
    marker=dict(size=6, color=filtered_df['Magnitude'], colorscale='Viridis', showscale=True, colorbar_title="Magnitude"),
    text=filtered_df['Country']
)])
fig_3d.update_layout(scene=dict(xaxis_title='Longitude', yaxis_title='Latitude', zaxis_title='Depth (km)'), title="3D View (Depth vs Location)")
st.plotly_chart(fig_3d, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Detailed Earthquake Data with Download
st.markdown("<div class='card'><h2 class='subheader'>Detailed Earthquake Data</h2>", unsafe_allow_html=True)
st.dataframe(filtered_df[['Country', 'Latitude', 'Longitude', 'Magnitude', 'Depth']].head(1000), use_container_width=True)
csv = filtered_df.to_csv(index=False)
st.download_button("Download Filtered Data", csv, "filtered_earthquakes.csv", "text/csv", key='download-csv')
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
<p>Built with Streamlit and Plotly. Data: Central Asia Earthquake 2019-2023 from <a href='https://www.kaggle.com/datasets/sitbayevalibek/central-asian-earthquake-dataset?resource=download' target='_blank'>Kaggle</a>.</p>
</div>
""", unsafe_allow_html=True)