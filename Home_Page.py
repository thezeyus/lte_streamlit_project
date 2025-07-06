import streamlit as st

# Set Streamlit app configuration
st.set_page_config(
    page_title="LTE Network Analyzer",
    page_icon="📶",
)

st.title("📡 LTE Network Analysis Dashboard")

st.markdown("""
Welcome to the LTE Network Analyzer. This multi-page application allows you to:

- 🔍 Explore LTE network KPI data (RSRP, RSRQ, SNR, etc.)
- 🧠 Apply machine learning models to classify and evaluate network quality
- 🗺️ Visualize serving cell coverage and performance on maps (coming soon)

Use the sidebar to navigate between pages. Make sure `LTE combined data.csv` is located in the root project folder.

---
💡 Tip: You can expand/collapse content on each page for a cleaner interface.

Ready? Start exploring from the **📊 Data Overview** page!
""")
