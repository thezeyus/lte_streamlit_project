import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Data Overview")
st.title("LTE Network Data Overview")

st.markdown("""
This dashboard provides a detailed exploration of LTE network quality metrics, known as **Key Performance Indicators (KPIs)**. Below is a brief explanation of each key metric:

- **RSRP (Reference Signal Received Power)**: Measures signal strength from a single cell. Lower (more negative) values indicate weaker signal. (unit: dBm)
- **RSRQ (Reference Signal Received Quality)**: Combines signal strength and interference. Useful for quality-based decisions. (unit: dB)
- **SNR (Signal-to-Noise Ratio)**: Describes how clean the signal is from noise. Higher is better. (unit: dB)
- **CQI (Channel Quality Indicator)**: Reported by the user equipment (UE), indicating channel conditions. Higher values mean better modulation/coding rates.
- **DL_bitrate / UL_bitrate**: Download and upload throughput as experienced by the user device. (unit: kbit/s)

These metrics collectively influence coverage, capacity, handover, and user quality of experience (QoE).
""")

# Load dataset
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "../LTE combined data.csv")
    df = pd.read_csv(file_path, low_memory=False)

    # Clean and convert data types
    for col in ["RSRP", "RSRQ", "SNR", "RSSI", "CQI", "DL_bitrate", "UL_bitrate", "ServingCell_Distance"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.dropna(subset=["RSRP", "RSRQ", "SNR", "DL_bitrate", "Speed"])

df = load_data()

# Use 'path' column as Mobility Pattern
mobility_col = "path"

# --- Distributions ---
st.header("🔍 Feature Distributions")
ignore_cols = ["Latitude", "Longitude", "ServingCell_Lat", "ServingCell_Lon"]
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(ignore_cols)
selected_col = st.selectbox("Select a feature to view distribution", numeric_cols)
fig, ax = plt.subplots()
sns.histplot(df[selected_col], kde=True, ax=ax)
ax.set_title(f"Distribution of {selected_col}")
st.pyplot(fig)

# --- Correlation Heatmap ---
st.header("📈 Feature Correlation")
with st.expander("Show correlation heatmap and insights"):
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    st.markdown("""
    **Correlation Insights:**
    - `RSRP` and `RSRQ` tend to be **positively correlated**: This makes sense since better signal strength often leads to better signal quality.
    - `CQI` correlates **moderately** with `DL_bitrate`: This indicates that better channel conditions often result in higher download speeds.
    - `SNR` is often **correlated** with both `CQI` and throughput — useful when modeling.
    - `RSSI` may correlate with `RSRP` but includes more interference, thus correlation may be weaker.
    """)

# --- Context-Aware Plots ---
st.header("🧠 Context-Aware Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("CQI vs DL_bitrate")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x="CQI", y="DL_bitrate", alpha=0.5, ax=ax3)
    st.pyplot(fig3)

with col2:
    st.subheader("Speed vs DL_bitrate")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=df, x="Speed", y="DL_bitrate", alpha=0.5, ax=ax4)
    st.pyplot(fig4)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Distance vs RSRP")
    fig5, ax5 = plt.subplots()
    sns.scatterplot(data=df, x="ServingCell_Distance", y="RSRP", alpha=0.5, ax=ax5)
    st.pyplot(fig5)

with col4:
    st.subheader("Distance vs RSRQ")
    fig6, ax6 = plt.subplots()
    sns.scatterplot(data=df, x="ServingCell_Distance", y="RSRQ", alpha=0.5, ax=ax6)
    st.pyplot(fig6)

st.subheader("Distance vs SNR")
fig7, ax7 = plt.subplots()
sns.scatterplot(data=df, x="ServingCell_Distance", y="SNR", alpha=0.5, ax=ax7)
st.pyplot(fig7)

# --- KPI Summary by Mobility Pattern ---
st.header("📊 Average KPIs by Mobility Pattern")
kpi_cols = ["RSRP", "RSRQ", "SNR", "CQI", "DL_bitrate"]
kpi_summary = df.groupby(mobility_col)[kpi_cols].mean().reset_index()
st.dataframe(kpi_summary, use_container_width=True)

for col in kpi_cols:
    fig, ax = plt.subplots()
    sns.barplot(data=kpi_summary, x=mobility_col, y=col, ax=ax)
    ax.set_title(f"Average {col} by Mobility Pattern")
    st.pyplot(fig)

st.success("Analysis complete! Explore more in other pages 🔍")
