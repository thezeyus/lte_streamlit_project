import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif, SelectKBest

st.set_page_config(page_title="Network Mode Prediction", layout="wide")
st.title("📡 Predicting Network Mode with Feature Selection")

@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "../LTE combined data.csv")
    df = pd.read_csv(file_path, low_memory=False)
    # Convert KPI columns to numeric
    kpi_cols = ["RSRP", "RSRQ", "SNR", "RSSI", "CQI", "DL_bitrate", "UL_bitrate", "ServingCell_Distance", "Speed"]
    for col in kpi_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill missing KPI values with median to retain all network modes
        df[col].fillna(df[col].median(), inplace=True)
    # Drop rows without NetworkMode
    df = df.dropna(subset=["NetworkMode"])
    return df

df = load_data()

# Show available network modes
st.subheader("NetworkMode Distribution")
mode_counts = df['NetworkMode'].value_counts()
st.bar_chart(mode_counts)
st.write(mode_counts.to_frame(name='Count'))

# Encode target
le = LabelEncoder()
df['NetworkMode_enc'] = le.fit_transform(df['NetworkMode'])

# Compute numeric features
numeric_feats = ["RSRP", "RSRQ", "SNR", "RSSI", "CQI", "DL_bitrate", "UL_bitrate", "ServingCell_Distance", "Speed"]

# Correlation with encoded target
def compute_correlations(df, features, target):
    corr_values = {}
    for f in features:
        corr = np.corrcoef(df[f], df[target])[0,1]
        corr_values[f] = corr
    return pd.Series(corr_values).sort_values(key=lambda x: x.abs(), ascending=False)

st.subheader("Feature Correlation with Network Mode")
corr_series = compute_correlations(df, numeric_feats, 'NetworkMode_enc')
fig_corr, ax_corr = plt.subplots(figsize=(8,4))
sns.barplot(x=corr_series.values, y=corr_series.index, palette='viridis', ax=ax_corr)
ax_corr.set_title('Correlation with NetworkMode')
ax_corr.set_xlabel('Pearson Correlation Coefficient')
st.pyplot(fig_corr)

st.markdown("""
**Insight:** Correlations show which KPIs vary most between 2G, 3G, and LTE modes.
""")

# Mutual Information ranking
st.subheader("Mutual Information Feature Ranking")
X = df[numeric_feats]
y = df['NetworkMode_enc']
mi = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi, index=numeric_feats).sort_values(ascending=False)
fig_mi, ax_mi = plt.subplots(figsize=(8,4))
sns.barplot(x=mi_series.values, y=mi_series.index, palette='magma', ax=ax_mi)
ax_mi.set_title('Mutual Information Scores')
ax_mi.set_xlabel('Mutual Information')
st.pyplot(fig_mi)

# Feature selection
st.subheader("Select Top K Features for Modeling")
k = st.slider('Number of features to select', min_value=1, max_value=len(numeric_feats), value=5)
selector = SelectKBest(mutual_info_classif, k=k)
X_selected = selector.fit_transform(X, y)
best_features = [feat for feat, mask in zip(numeric_feats, selector.get_support()) if mask]
st.markdown(f"**Top {k} features:** {best_features}")

# Modeling with selected features
st.subheader("Model Performance with Selected Features")
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Compute classes present in test set
test_labels = np.unique(y_test)
test_class_names = [le.classes_[i] for i in test_labels]

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred, labels=test_labels, target_names=test_class_names))

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred, labels=test_labels)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_class_names, yticklabels=test_class_names, ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
st.pyplot(fig_cm)

st.success("Feature selection and modeling complete for all NetworkMode values!")
