import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Mobility & Handover Insights", layout="wide")
st.title("🚦 Mobility & Handover Insights")

@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "../LTE combined data.csv")
    df = pd.read_csv(file_path, low_memory=False)
    # Convert KPIs to numeric
    kpi_cols = ["RSRP","RSRQ","SNR","RSSI","CQI","DL_bitrate","UL_bitrate","ServingCell_Distance","Speed"]
    for col in kpi_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Parse timestamps
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%Y.%m.%d_%H.%M.%S", errors='coerce')
    # Drop rows without key info
    df = df.dropna(subset=["Timestamp","CellID"] + kpi_cols)
    return df.sort_values('Timestamp')

df = load_data()

# --- Mobility Pattern Overview ---
st.subheader("Mobility Pattern Distribution and Speed")
mode_counts = df['path'].value_counts()
st.bar_chart(mode_counts)
st.write(mode_counts.to_frame(name='Count'))

fig_speed, ax_speed = plt.subplots(figsize=(8,4))
sns.boxplot(x='path', y='Speed', data=df, ax=ax_speed)
ax_speed.set_title('Speed Distribution by Mobility Pattern')
st.pyplot(fig_speed)

# --- Handover Event Detection ---
st.subheader("Handover Event Detection and Insights")
df['prev_CellID'] = df['CellID'].shift(1)
df['handover'] = np.where(df['CellID'] != df['prev_CellID'], 1, 0)
handover_counts = df.groupby('path')['handover'].sum()
st.write("Number of handover events per mobility pattern:")
st.bar_chart(handover_counts)

st.markdown("""
**Handover & Mobility Insight:**  
- **Static:** Very few handovers due to no movement.  
- **Pedestrian:** Moderate handovers as users walk across small cell boundaries.  
- **Bus/Car:** Increased handovers reflecting higher speeds and frequent cell transitions.  
- **Train:** Highest handover counts, consistent with rapid travel across cell coverage areas.
""")

# --- Time Series Inspection for All Patterns ---
st.subheader("Time Series of RSRP with Handover Markers for Each Pattern")
patterns = df['path'].unique().tolist()
tabs = st.tabs(patterns)
for tab, pattern in zip(tabs, patterns):
    with tab:
        df_pat = df[df['path'] == pattern]
        # Plot RSRP time series
        fig_ts, ax_ts = plt.subplots(figsize=(12,4))
        ax_ts.plot(df_pat['Timestamp'], df_pat['RSRP'], label='RSRP')
        # Mark handovers
        ho = df_pat[df_pat['handover'] == 1]
        ax_ts.scatter(ho['Timestamp'], ho['RSRP'], color='red', label='Handover', s=20)
        ax_ts.set_ylabel('RSRP (dBm)')
        ax_ts.set_xlabel('Time')
        ax_ts.set_title(f'RSRP & Handovers ({pattern})')
        ax_ts.legend()
        st.pyplot(fig_ts)
        # Insight comment per pattern
        st.markdown(f"**{pattern.capitalize()} Insight:** Speed median {df_pat['Speed'].median():.1f} km/h, total handovers {int(handover_counts[pattern])}. "
                    + "Rapid movements correspond to spikes in handover markers.")
        # Plot CellID over time
        fig_cid, ax_cid = plt.subplots(figsize=(12,3))
        ax_cid.step(df_pat['Timestamp'], df_pat['CellID'], where='post')
        ax_cid.set_ylabel('CellID')
        ax_cid.set_xlabel('Time')
        ax_cid.set_title(f'CellID Transitions ({pattern})')
        st.pyplot(fig_cid)

st.success("Mobility & Handover Insights complete! Explore map visualizations next.")
