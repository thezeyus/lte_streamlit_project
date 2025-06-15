import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ML Modeling")
st.title("🤖 Machine Learning Modeling")

# Load and clean data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "../LTE combined data.csv")
    df = pd.read_csv(file_path, low_memory=False)
    for col in ["RSRP", "RSRQ", "SNR", "RSSI", "CQI", "DL_bitrate", "UL_bitrate", "ServingCell_Distance"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=["RSRP", "RSRQ", "SNR", "DL_bitrate", "Speed", "path"])
    return df

df = load_data()

# Let user select a target
target_options = {
    "Mobility Pattern (path)": "path",
    "CQI Level (Binned)": "CQI_binned",
    "Download Bitrate (Binned)": "DL_bitrate_binned"
}

# Add binned targets
df["CQI_binned"] = pd.qcut(df["CQI"], q=3, labels=["Low", "Mid", "High"])
df["DL_bitrate_binned"] = pd.qcut(df["DL_bitrate"], q=3, labels=["Low", "Mid", "High"])

target_label = st.selectbox("Select Target Variable", list(target_options.keys()))
target = target_options[target_label]

# Feature selection
features = ["RSRP", "RSRQ", "SNR", "CQI", "DL_bitrate", "UL_bitrate", "RSSI", "ServingCell_Distance"]
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)

st.header("📊 Model Performance")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# --- Feature Importances ---
st.subheader("🔍 Feature Importances")
importances = model.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
fig2, ax2 = plt.subplots()
sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax2)
ax2.set_title("Feature Importance (Random Forest)")
st.pyplot(fig2)

st.success("Modeling complete! You can try different targets or tune the model further.")
