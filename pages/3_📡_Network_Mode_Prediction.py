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

"""
📡 Network Mode Prediction
-------------------------
Compare two modelling strategies to show whether **Stratified CV + SMOTE**
helps recognise minority network modes.

Pipelines compared
~~~~~~~~~~~~~~~~~~
1. **Baseline** – stratified train‑test split, *no* resampling.
2. **Stratified CV + SMOTE** – oversample minorities *inside* each fold,
   then train on the full train set with SMOTE.

Outputs: per‑class metrics, confusion matrices, and a macro‑metric bar‑chart.

🔧 *Note* `st.set_page_config()` is **not** called here to avoid the
"can only be called once" error. Configure the app once in `Home_Page.py`.
"""

# ---------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------

st.title("📡 Network Mode Prediction – Baseline vs Stratified CV + SMOTE")

st.markdown(
    """
    **Goal** Demonstrate whether class‑imbalance aware modelling really improves
    the ability to recognise minority network modes (e.g. EDGE, HSUPA) compared
    with a regular model.
    
    **Pipelines**
    1. **Baseline** – simple stratified split, **no** oversampling.
    2. **Stratified CV + SMOTE** – 5‑fold stratified cross‑validation with
       SMOTE in each fold, followed by a final model trained on SMOTE‑balanced
       data.
    """
)

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_dataset():
    """Load LTE combined data and perform basic cleaning."""
    path = os.path.join(os.path.dirname(__file__), "../LTE combined data.csv")
    df = pd.read_csv(path, low_memory=False)
    kpis = [
        "RSRP",
        "RSRQ",
        "SNR",
        "RSSI",
        "CQI",
        "DL_bitrate",
        "UL_bitrate",
        "ServingCell_Distance",
        "Speed",
    ]
    for c in kpis:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c].fillna(df[c].median(), inplace=True)
    return df.dropna(subset=["NetworkMode"]).copy()


def train_baseline(X_train, X_test, y_train, y_test, labels, target_names):
    scaler = StandardScaler().fit(X_train)
    X_tr_s, X_te_s = scaler.transform(X_train), scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr_s, y_train)
    preds = model.predict(X_te_s)
    rpt = classification_report(
        y_test,
        preds,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=labels)
    return rpt, cm


def train_smote_cv(X_train, y_train, labels, skf):
    reports = []
    agg_cm = np.zeros((len(labels), len(labels)), dtype=int)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        min_count = y_tr.value_counts().min()
        if min_count > 1:
            k = min(5, min_count - 1)
            sm = SMOTE(random_state=42, k_neighbors=k)
            X_res, y_res = sm.fit_resample(X_tr, y_tr)
        else:
            st.warning(
                f"Fold {fold}: minority class has only {min_count} sample(s); skipping SMOTE"
            )
            X_res, y_res = X_tr, y_tr
        scaler = StandardScaler().fit(X_res)
        X_res_s, X_val_s = scaler.transform(X_res), scaler.transform(X_val)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_res_s, y_res)
        preds = clf.predict(X_val_s)
        rpt = classification_report(
            y_val,
            preds,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        reports.append(pd.DataFrame(rpt).T)
        agg_cm += confusion_matrix(y_val, preds, labels=labels)
    return reports, agg_cm


def train_final_smote(X_train, y_train):
    min_full = y_train.value_counts().min()
    if min_full > 1:
        k = min(5, min_full - 1)
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X_train, y_train)
    else:
        st.warning(
            f"Full train: minority class has only {min_full} sample(s); skipping SMOTE"
        )
        X_res, y_res = X_train, y_train
    scaler = StandardScaler().fit(X_res)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(scaler.transform(X_res), y_res)
    return clf, scaler

# ---------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------

df = load_dataset()
le = LabelEncoder()
df["mode_enc"] = le.fit_transform(df["NetworkMode"])
features = [
    "RSRP",
    "RSRQ",
    "SNR",
    "RSSI",
    "CQI",
    "DL_bitrate",
    "UL_bitrate",
    "ServingCell_Distance",
    "Speed",
]
X = df[features]
y = df["mode_enc"]
labels = list(range(len(le.classes_)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Training distribution
st.subheader("🔎 Training‑set Class Distribution")
train_dist = y_train.value_counts().rename(index=dict(enumerate(le.classes_)))
col1, col2 = st.columns([2, 1])
with col1:
    st.bar_chart(train_dist)
with col2:
    st.write(train_dist.to_frame(name="count"))

# ---------------------------------------------------------------------
# Modelling Tabs
# ---------------------------------------------------------------------

tabs = st.tabs(["🔹 Baseline", "🔸 Stratified CV + SMOTE"])

with tabs[0]:
    st.header("🔹 Baseline – No Resampling")
    base_report, base_cm = train_baseline(
        X_train, X_test, y_train, y_test, labels, le.classes_
    )
    rpt_df = pd.DataFrame(base_report).T.round(3)
    st.dataframe(rpt_df[["precision", "recall", "f1-score", "support"]])
    fig_b, ax_b = plt.subplots()
    sns.heatmap(
        base_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_b,
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    ax_b.set_xlabel("Predicted")
    ax_b.set_ylabel("Actual")
    st.pyplot(fig_b)

with tabs[1]:
    st.header("🔸 Stratified CV + SMOTE")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_reports, cv_cm = train_smote_cv(X_train, y_train, labels, skf)
    if cv_reports:
        avg_cv = pd.concat(cv_reports).groupby(level=0).mean().round(3)
        st.subheader("Cross‑validation Metrics (mean of 5 folds)")
        st.dataframe(avg_cv[["precision", "recall", "f1-score", "support"]])
        fig_cv, ax_cv = plt.subplots()
        sns.heatmap(
            cv_cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            ax=ax_cv,
            xticklabels=le.classes_,
            yticklabels=le.classes_,
        )
        ax_cv.set_title("Aggregated CV Confusion Matrix")
        st.pyplot(fig_cv)

    st.subheader("Final Model on Test Set")
    final_clf, final_scaler = train_final_smote(X_train, y_train)
    preds_test = final_clf.predict(final_scaler.transform(X_test))
    smote_report = classification_report(
        y_test,
        preds_test,
        labels=labels,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    rpt_smote_df = pd.DataFrame(smote_report).T.round(3)
    st.dataframe(rpt_smote_df[["precision", "recall", "f1-score", "support"]])
    cm_smote = confusion_matrix(y_test, preds_test, labels=labels)
    fig_sm, ax_sm = plt.subplots()
    sns.heatmap(
        cm_smote,
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax_sm,
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    ax_sm.set_xlabel("Predicted")
    ax_sm.set_ylabel("Actual")
    st.pyplot(fig_sm)

# ---------------------------------------------------------------------
# Summary Comparison Table
# ---------------------------------------------------------------------

st.header("📊 Macro‑level Comparison (Test Set)")
base_macro = {
    "precision": base_report["macro avg"]["precision"],
    "recall": base_report["macro avg"]["recall"],
    "f1": base_report["macro avg"]["f1-score"],
}
smote_macro = {
    "precision": smote_report["macro avg"]["precision"],
    "recall": smote_report["macro avg"]["recall"],
    "f1": smote_report["macro avg"]["f1-score"],
}
summary_df = (
    pd.DataFrame({"Baseline": base_macro, "SMOTE": smote_macro})
    .T
    .rename_axis("Model")
    .reset_index()
)
summary_df[["precision", "recall", "f1"]] = summary_df[["precision", "recall", "f1"]].round(3)

col_sum1, col_sum2 = st.columns([1, 2])
with col_sum1:
    st.dataframe(summary_df)
with col_sum2:
    fig_bar, ax_bar = plt.subplots()
    summary_df_melt = summary_df.melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    sns.barplot(data=summary_df_melt, x="Metric", y="Score", hue="Model", ax=ax_bar)
    ax_bar.set_ylim(0, 1)
    st.pyplot(fig_bar)

st.success("Comparison complete – check whether macro‑F1 improved with SMOTE!")
