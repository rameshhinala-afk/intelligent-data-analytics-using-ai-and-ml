"""Intelligent Data Analytics Using AI and ML - Streamlit app.
Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

import analytics_engine as ae
from sample_data import make_sample

st.set_page_config(page_title="Intelligent Data Analytics", page_icon="📊", layout="wide")
st.title("📊 Intelligent Data Analytics Using AI & ML")
st.caption("Upload a CSV and get automatic profiling, cleaning, AutoML, clustering and anomaly detection.")

# ---- Data input ----
with st.sidebar:
    st.header("Data")
    file = st.file_uploader("Upload CSV", type="csv")
    use_sample = st.checkbox("Use sample customer data", value=file is None)

if file is not None:
    raw = pd.read_csv(file)
elif use_sample:
    raw = make_sample()
else:
    st.info("Upload a CSV or tick the sample data box in the sidebar.")
    st.stop()

tabs = st.tabs(["🔍 Overview", "🧹 Cleaning", "🤖 AutoML", "🧩 Clustering", "🚨 Anomalies"])

# ---- Overview ----
with tabs[0]:
    p = ae.profile(raw)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{p['rows']:,}")
    c2.metric("Columns", p["cols"])
    c3.metric("Missing cols", len(p["missing"]))
    c4.metric("Duplicates", p["duplicates"])
    st.subheader("💡 Automatic insights")
    for line in ae.generate_insights(raw):
        st.write("•", line)
    st.subheader("Data preview")
    st.dataframe(raw.head(50), width="stretch")
    if not p["summary"].empty:
        st.subheader("Numeric summary")
        st.dataframe(p["summary"], width="stretch")
        col = st.selectbox("Distribution of", p["numeric_cols"])
        counts = pd.cut(raw[col].dropna(), bins=20).value_counts().sort_index()
        st.bar_chart(pd.Series(counts.values, index=counts.index.astype(str), name="count"))

# ---- Cleaning ----
cleaned, log = ae.clean(raw)
with tabs[1]:
    st.subheader("Cleaning report")
    for line in log:
        st.write("✅", line)
    st.write(f"Shape: {raw.shape} → {cleaned.shape}")
    st.download_button("Download cleaned CSV", cleaned.to_csv(index=False), "cleaned.csv", "text/csv")

# ---- AutoML ----
with tabs[2]:
    target = st.selectbox("Target column to predict", cleaned.columns,
                          index=len(cleaned.columns) - 1)
    if st.button("Run AutoML", type="primary"):
        with st.spinner("Training and comparing models…"):
            res = ae.auto_ml(cleaned, target)
        st.success(f"Task: **{res['task']}** · Best model: **{res['best_model']}**")
        cols = st.columns(len(res["metrics"]))
        for col, (k, v) in zip(cols, res["metrics"].items()):
            col.metric(k, v)
        st.subheader("Model leaderboard")
        st.dataframe(res["leaderboard"], width="stretch")
        st.subheader("Feature importance (permutation)")
        st.bar_chart(res["importance"].set_index("Feature"))

# ---- Clustering ----
with tabs[3]:
    auto_k = st.checkbox("Auto-select number of clusters", value=True)
    k = None if auto_k else st.slider("Clusters (k)", 2, 8, 3)
    if st.button("Run clustering"):
        try:
            res = ae.cluster(cleaned, k)
            st.success(f"k = {res['k']} · silhouette score = {res['silhouette']}")
            st.scatter_chart(res["plot"], x="PC1", y="PC2", color="Cluster")
            st.subheader("Cluster profiles (mean values)")
            st.dataframe(res["profile"], width="stretch")
        except ValueError as e:
            st.error(str(e))

# ---- Anomalies ----
with tabs[4]:
    cont = st.slider("Expected anomaly share", 0.01, 0.20, 0.05, 0.01)
    if st.button("Detect anomalies"):
        res = ae.detect_anomalies(cleaned, cont)
        st.write(f"Found **{int(res['is_anomaly'].sum())}** anomalous rows.")
        st.dataframe(res[res["is_anomaly"]].head(100), width="stretch")
