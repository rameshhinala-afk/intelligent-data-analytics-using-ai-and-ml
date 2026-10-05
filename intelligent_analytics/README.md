# Intelligent Data Analytics Using AI and ML

Upload any CSV and the app automatically profiles it, cleans it, trains and compares ML models,
finds customer-like segments, and flags anomalies.

| Module | Technique |
|---|---|
| Profiling & insights | Descriptive stats, correlation, skewness rules |
| Cleaning | Duplicate removal, median/mode imputation |
| AutoML | Auto task detection, pipeline preprocessing, 3-model CV comparison, permutation importance |
| Clustering | KMeans with silhouette-based k selection, PCA visualisation |
| Anomaly detection | Isolation Forest |

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – Streamlit UI
- `analytics_engine.py` – all analytics/ML logic (usable without the UI)
- `sample_data.py` – demo dataset generator (`python sample_data.py` writes a CSV)

## Ideas to extend
SHAP explanations, time-series forecasting, hyperparameter tuning (Optuna), model download (joblib), LLM-generated report.
