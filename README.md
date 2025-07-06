## Final Project

The Project is deployed on Streamlit Cloud, you can interact with the project with [this link](https://lteapp.streamlit.app/).

Or you can run the dashboard locally: 

1. Clone the repository:
   ```bash
   git clone https://github.com/thezeyus/lte_streamlit_project.git
   cd lte_streamlit_project
   
2. Install the required libraries:

   ```bash
   pip install -r requirements.txt

3. Launch the Streamlit app:

   ```bash
   python -m streamlit run streamlit_app.py


### Features
**Data Overview**: Visual exploration of LTE KPIs (RSRP, RSRQ, SNR, CQI, etc.) with correlation analysis.

**Machine Learning Models**: Predict mobility patterns and network modes using Random Forests.

**Network Mode Prediction**: Classifies 2G / 3G / 4G with explanation of feature importance.

**Mobility & Handover Insights**: Handover frequency and signal patterns across different mobility types.

**Interactive Maps**: View device locations and serving cell performance with KPI-based coloring.

**Cell Location Proposal**: Recommends new cell sites using K-Means clustering over poor-performance areas.

### Dataset

This dashboard is powered by LTE drive-test data collected around Cork City, with KPIs like RSRP, RSRQ, SNR, CQI, throughput, and distance from serving cells. Mobility patterns include Static, Pedestrian, Bus, Train, and Car scenarios. You can access the dataset via [this link](https://www.kaggle.com/datasets/aeryss/lte-dataset).



