import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pydeck as pdk
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

st.set_page_config(page_title="🏗️ Cell Placement Proposal", layout="wide")
st.title("🏗️ Cell Location Investment Proposal")

st.markdown("""
This page suggests new cell tower locations to improve network coverage based on the poorest-performing areas, separated by mobility pattern.

**Approach per pattern**:
1. Identify low-performance samples per pattern.
2. Cluster these samples using K-Means.
3. Evaluate coverage improvement within a target radius.

You can adjust KPI thresholds, radius, and number of proposed cells independently for each pattern.
""")

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../LTE combined data.csv'), low_memory=False)
    # Ensure numeric values
    for col in ['Latitude','Longitude','RSRP','RSRQ','SNR']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['path'] = df['path'].fillna('Unknown')
    return df.dropna(subset=['Latitude','Longitude','RSRP','RSRQ','SNR','path'])

df = load_data()
patterns = df['path'].unique().tolist()

tabs = st.tabs(patterns)
for tab, pattern in zip(tabs, patterns):
    with tab:
        st.header(f"🚩 Mobility Pattern: {pattern}")
        df_pat = df[df['path'] == pattern].copy()
        # Per-pattern inputs
        kpi = st.selectbox("Select KPI for worst-area detection", ['RSRP','RSRQ','SNR'], key=f"kpi_{pattern}")
        thr_default = float(df_pat[kpi].quantile(0.1))
        threshold = st.slider("Threshold for poor performance", float(df_pat[kpi].min()), float(df_pat[kpi].max()), thr_default, key=f"thr_{pattern}")
        radius = st.slider("Coverage radius (meters)", 100, 2000, 500, key=f"rad_{pattern}")
        n_cells = st.slider("Number of new cell sites to propose", 1, 10, 3, key=f"ncell_{pattern}")

        # Identify worst-performing samples
        worst = df_pat[df_pat[kpi] <= threshold][['Latitude','Longitude']].rename(columns={'Latitude':'lat','Longitude':'lon'})
        st.write(f"{pattern}: {len(worst)} poor samples (KPI {kpi} ≤ {threshold})")

        if len(worst) == 0:
            st.warning("No poor samples for this pattern with current threshold.")
            continue

        # Clustering
        km = KMeans(n_clusters=n_cells, random_state=42).fit(worst)
        centers = pd.DataFrame(km.cluster_centers_, columns=['lat','lon'])

        # Coverage calculation
        dists, _ = pairwise_distances_argmin_min(worst, centers)
        covered = ((dists * 111000) <= radius).sum()
        pct = covered / len(worst) * 100
        st.write(f"Coverage: {covered}/{len(worst)} points ({pct:.1f}%) within {radius}m radius")

        # Map of poor samples & proposed sites
        st.subheader("Map: Poor Samples & Proposed Sites")
        layer_poor = pdk.Layer(
            "ScatterplotLayer", data=worst,
            get_position=['lon','lat'], get_color=[255,0,0,80], get_radius=50
        )
        layer_centers = pdk.Layer(
            "ScatterplotLayer", data=centers,
            get_position=['lon','lat'], get_color=[255,255,0,200], get_radius=100
        )
        mid_lat = df_pat['Latitude'].median()
        mid_lon = df_pat['Longitude'].median()
        view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=11)
        deck = pdk.Deck(layers=[layer_poor, layer_centers], initial_view_state=view,
                        map_style='mapbox://styles/mapbox/light-v10',
                        tooltip={"text": "Center: {lat}, {lon}"})
        st.pydeck_chart(deck)

        # Coverage improvement plot
        st.subheader("Coverage Improvement vs # of New Sites")
        results = []
        for k in range(1, n_cells + 1):
            km2 = KMeans(n_clusters=k, random_state=42).fit(worst)
            ctrs = km2.cluster_centers_
            d2, _ = pairwise_distances_argmin_min(worst, ctrs)
            cov = ((d2 * 111000) <= radius).sum()
            results.append({'New Sites': k, 'Coverage (%)': cov / len(worst) * 100})
        rdf = pd.DataFrame(results)
        fig, ax = plt.subplots()
        ax.plot(rdf['New Sites'], rdf['Coverage (%)'], marker='o')
        ax.set_xlabel('Number of New Sites')
        ax.set_ylabel('Coverage of Poor Samples (%)')
        ax.set_title(f'{pattern} Coverage vs # of Sites')
        st.pyplot(fig)

st.success("Per-pattern cell placement proposal complete!")
