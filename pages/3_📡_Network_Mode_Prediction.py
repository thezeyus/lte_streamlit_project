import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Network Mode Prediction", layout="wide")
st.title("📡 Predicting Network Mode with Robust CV and Oversampling")

# --- Descriptions for clarity ---
st.subheader("Network Mode Descriptions")
# Brief summary
st.markdown(
    """
- **2G (GSM/EDGE):** Voice-focused, low-speed data (~0.1 Mbps).
- **3G (UMTS/HSPA):** Medium-speed, supports multimedia (several Mbps).
- **4G LTE:** High-speed broadband (tens–hundreds Mbps) via OFDM.
    """
)
# Expanders for each mode
with st.expander("2G (GSM/EDGE) Details", expanded=False):
    st.markdown(
        """
2G introduced digital voice encoding and basic data via GPRS (up to ~114 kbps) and EDGE (up to ~384 kbps).
It laid the foundation for SMS, basic internet, and roaming across multiple carriers.
        """
    )
with st.expander("3G (UMTS/HSPA) Details", expanded=False):
    st.markdown(
        """
**UMTS (3G):** Utilizes W-CDMA radio access for higher data rates (up to ~2 Mbps).\
**HSPA:** Adds HSDPA/HSUPA for peak downlink/up to ~14/5.76 Mbps.\

**HSPA+:** Evolved HSPA with 64‑QAM and MIMO, pushing downlink to ~42 Mbps+, lower latency, and better spectral efficiency.
        """
    )
with st.expander("4G (LTE) Details", expanded=False):
    st.markdown(
        """
LTE employs OFDM for downlink and SC-FDMA for uplink, achieving tens to hundreds of Mbps.\
Supports VoLTE (voice over LTE), carrier aggregation, and reduced latency (<10 ms), enabling real-time applications.
        """
    )

@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "../LTE combined data.csv")
    df = pd.read_csv(path, low_memory=False)
    kpis = ["RSRP","RSRQ","SNR","RSSI","CQI","DL_bitrate","UL_bitrate","ServingCell_Distance","Speed"]
    for c in kpis:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c].fillna(df[c].median(), inplace=True)
    return df.dropna(subset=["NetworkMode"])

# Load and encode data
_df = load_data()
le = LabelEncoder()
_df['NetworkMode_enc'] = le.fit_transform(_df['NetworkMode'])
features = ["RSRP","RSRQ","SNR","RSSI","CQI","DL_bitrate","UL_bitrate","ServingCell_Distance","Speed"]
X = _df[features]
y = _df['NetworkMode_enc']
n_classes = len(le.classes_)
labels = list(range(n_classes))

# 1) Split data first — hold out test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

# Training distribution
st.subheader("Train Set Network Mode Distribution")
train_counts = pd.Series(y_train).map({i:cls for i,cls in enumerate(le.classes_)}).value_counts()
st.bar_chart(train_counts)
st.write(train_counts.to_frame(name='count'))

# 2) Correlation on training KPIs
st.subheader("KPI Correlation Matrix (Train)")
corr = X_train.corr()
fig1, ax1 = plt.subplots(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax1)
ax1.set_title('Correlation Matrix of KPIs (Train)')
st.pyplot(fig1)

# 3) Stratified CV with SMOTE per fold
st.subheader("Stratified CV + SMOTE Oversampling")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics_dfs = []
aggregate_cm = None
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    counts = y_tr.value_counts()
    min_count = counts.min()
    if min_count > 1:
        k_neighbors = min(5, min_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X_tr, y_tr)
    else:
        st.warning(f"Fold {fold}: minority class has only {min_count} sample(s), skipping SMOTE")
        X_res, y_res = X_tr, y_tr
    scaler_fold = StandardScaler().fit(X_res)
    X_res_s = scaler_fold.transform(X_res)
    X_val_s = scaler_fold.transform(X_val)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_res_s, y_res)
    y_pred = clf.predict(X_val_s)
    rpt_dict = classification_report(y_val, y_pred, output_dict=True)
    df_rpt = pd.DataFrame(rpt_dict).T
    metrics_dfs.append(df_rpt)
    cm_fold = confusion_matrix(y_val, y_pred, labels=labels)
    aggregate_cm = cm_fold if aggregate_cm is None else aggregate_cm + cm_fold

# Average CV metrics
if metrics_dfs:
    avg_metrics = pd.concat(metrics_dfs).groupby(level=0).mean()
    st.dataframe(avg_metrics[['precision','recall','f1-score','support']].round(3))

# Aggregated CV confusion matrix
if aggregate_cm is not None:
    fig_ag, ax_ag = plt.subplots(figsize=(6,6))
    sns.heatmap(aggregate_cm, annot=True, fmt='d',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_ag)
    ax_ag.set_title('Aggregated CV Confusion Matrix')
    st.pyplot(fig_ag)

# 4) Final evaluation on untouched test set
st.subheader("Final Test Set Performance")
full_counts = y_train.value_counts()
min_full = full_counts.min()
if min_full > 1:
    k_full = min(5, min_full - 1)
    smote_full = SMOTE(random_state=42, k_neighbors=k_full)
    X_full_res, y_full_res = smote_full.fit_resample(X_train, y_train)
else:
    st.warning(f"Full train: minority class has only {min_full} sample(s), skipping SMOTE on full train")
    X_full_res, y_full_res = X_train, y_train
scaler_full = StandardScaler().fit(X_full_res)
clf_full = RandomForestClassifier(n_estimators=100, random_state=42)
clf_full.fit(scaler_full.transform(X_full_res), y_full_res)
X_test_s = scaler_full.transform(X_test)
y_pred_test = clf_full.predict(X_test_s)
# Ensure labels parameter to match target_names
st.text(classification_report(y_test, y_pred_test, labels=labels, target_names=le.classes_))
cm_full = confusion_matrix(y_test, y_pred_test, labels=labels)
fig2, ax2 = plt.subplots()
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
st.pyplot(fig2)

st.success("Modeling with proper CV and oversampling completed.")
