import streamlit as st
import pandas as pd
import pydeck as pdk
import os

st.set_page_config(page_title="Map Visualization")
st.title("🗺️ Map Visualization")

st.markdown("""
This page visualizes the geographic context of our LTE dataset, collected around **Cork City, Ireland**. Cork is Ireland's second-largest city, featuring urban, suburban, and rural areas:

- **Urban**: dense cell placement for capacity and indoor coverage.
- **Suburban**: balanced coverage and capacity trade-offs.
- **Rural**: wide-area cells to cover sparse populations.

Select a mobility pattern and KPI to view cell performance categories on the map.
""")

@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "../LTE combined data.csv")
    df = pd.read_csv(file_path, low_memory=False)
    # Ensure numeric types for coordinates and KPIs
    for col in ['Latitude','Longitude','ServingCell_Lat','ServingCell_Lon','RSRP','RSRQ','SNR']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['path'] = df['path'].fillna('Unknown')
    return df.dropna(subset=['Latitude','Longitude','ServingCell_Lat','ServingCell_Lon','RSRP','RSRQ','SNR','path'])

df = load_data()
patterns = df['path'].unique().tolist()

# KPI categorization based on thresholds
def categorize_kpi(df_kpi, kpi):
    if kpi == 'RSRP':
        bins = [-999, -100, -90, -80, 999]
        labels = ['No signal', 'Fair to poor', 'Good', 'Excellent']
    elif kpi == 'RSRQ':
        bins = [-999, -20, -15, -10, 999]
        labels = ['No signal', 'Fair to poor', 'Good', 'Excellent']
    else:  # SNR
        bins = [-999, 0, 13, 20, 999]
        labels = ['No signal', 'Fair to poor', 'Good', 'Excellent']
    df_kpi['category'] = pd.cut(df_kpi[kpi], bins=bins, labels=labels)
    # RGBA colors
    color_map = {
        'Excellent': [0, 200, 0, 150],
        'Good': [173, 255, 47, 150],
        'Fair to poor': [255, 165, 0, 150],
        'No signal': [255, 0, 0, 150]
    }
    for idx, c in enumerate(['r','g','b','a']):
        df_kpi[c] = df_kpi['category'].map(lambda x: color_map[x][idx])
    return df_kpi

# Create tabs per mobility pattern
tabs = st.tabs(patterns)
for tab, pattern in zip(tabs, patterns):
    with tab:
        st.subheader(f"🚩 Mobility Pattern: {pattern}")
        df_pat = df[df['path'] == pattern].copy()
        # KPI selection
        kpi = st.selectbox("Select KPI for performance", ['RSRP','RSRQ','SNR'], key=pattern)
        # Aggregate KPI by cell
        cell_perf = df_pat.groupby(['ServingCell_Lat','ServingCell_Lon'])[kpi].mean().reset_index()
        cell_perf = categorize_kpi(cell_perf, kpi)
        # Device points
        device_df = df_pat.rename(columns={'Latitude':'lat','Longitude':'lon'})

        # Legend
        st.markdown("**Performance Categories:**")
        st.markdown(
            f"🟢 Excellent | {('≥ -80 dBm' if kpi=='RSRP' else ('≥ -10 dB' if kpi=='RSRQ' else '≥ 20 dB'))}<br>"
            + f"🟡 Good | {(' -90 to -80 dBm' if kpi=='RSRP' else (' -15 to -10 dB' if kpi=='RSRQ' else '13 to 20 dB'))}<br>"
            + f"🟠 Fair to poor | {(' -100 to -90 dBm' if kpi=='RSRP' else (' -20 to -15 dB' if kpi=='RSRQ' else '0 to 13 dB'))}<br>"
            + f"🔴 No signal | {('≤ -100 dBm' if kpi=='RSRP' else ('≤ -20 dB' if kpi=='RSRQ' else '< 0 dB'))}", unsafe_allow_html=True
        )

        # Map of towers & devices
        layer_towers = pdk.Layer(
            "ScatterplotLayer",
            data=cell_perf.rename(columns={'ServingCell_Lon':'lon','ServingCell_Lat':'lat'}),
            get_position=["lon","lat"],
            get_color=["r","g","b","a"],
            get_radius=100,
            pickable=True
        )
        layer_devices = pdk.Layer(
            "ScatterplotLayer",
            data=device_df[['lat','lon']],
            get_position=["lon","lat"],
            get_color=[0,0,255,50],
            get_radius=30,
            pickable=False
        )
        mid_lat = device_df['lat'].median()
        mid_lon = device_df['lon'].median()
        view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=11, pitch=45)
        tooltip_text = "{" + kpi + "}\n{category}"
        deck = pdk.Deck(
            layers=[layer_towers, layer_devices],
            initial_view_state=view,
            map_style="mapbox://styles/mapbox/light-v10",
            tooltip={"text": tooltip_text}
        )
        st.pydeck_chart(deck)

        # Device performance map
        st.subheader(f"Device Performance ({kpi}) for {pattern}")
        dev_perf = df_pat[['Latitude','Longitude',kpi]].copy()
        dev_perf.rename(columns={'Latitude':'lat','Longitude':'lon'}, inplace=True)
        dev_perf = categorize_kpi(dev_perf, kpi)
        layer_dev_perf = pdk.Layer(
            "ScatterplotLayer",
            data=dev_perf,
            get_position=["lon","lat"],
            get_color=["r","g","b","a"],
            get_radius=20,
            pickable=True
        )
        deck_dev = pdk.Deck(
            layers=[layer_dev_perf],
            initial_view_state=view,
            map_style="mapbox://styles/mapbox/light-v10",
            tooltip={"text": "{lat}, {lon}\n{category}"}
        )
        st.pydeck_chart(deck_dev)

st.success("Map visualization by KPI performance loaded!")
