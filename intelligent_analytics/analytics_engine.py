"""Core AI/ML analytics logic (independent of the UI so it can be reused/tested)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              IsolationForest, RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, r2_score,
                             silhouette_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------- 1. Profiling ----------
def profile(df: pd.DataFrame) -> dict:
    num = df.select_dtypes(include="number")
    cat = df.select_dtypes(exclude="number")
    missing = df.isna().sum()
    corr_pairs = []
    if num.shape[1] > 1:
        c = num.corr().abs()
        for i, a in enumerate(c.columns):
            for b in c.columns[i + 1:]:
                corr_pairs.append((a, b, round(float(num.corr().loc[a, b]), 3)))
        corr_pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return {
        "rows": len(df), "cols": df.shape[1],
        "numeric_cols": list(num.columns), "categorical_cols": list(cat.columns),
        "missing": missing[missing > 0].to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "top_correlations": corr_pairs[:5],
        "summary": num.describe().T.round(2) if not num.empty else pd.DataFrame(),
    }


def generate_insights(df: pd.DataFrame) -> list[str]:
    p = profile(df)
    out = [f"Dataset has {p['rows']:,} rows and {p['cols']} columns "
           f"({len(p['numeric_cols'])} numeric, {len(p['categorical_cols'])} categorical)."]
    if p["missing"]:
        worst = max(p["missing"], key=p["missing"].get)
        out.append(f"{len(p['missing'])} column(s) have missing values; worst is '{worst}' "
                   f"({p['missing'][worst] / p['rows']:.1%}).")
    if p["duplicates"]:
        out.append(f"{p['duplicates']} duplicate row(s) found.")
    for a, b, r in p["top_correlations"][:3]:
        if abs(r) >= 0.5:
            out.append(f"Strong {'positive' if r > 0 else 'negative'} correlation between '{a}' and '{b}' (r={r}).")
    for col in p["numeric_cols"]:
        skew = df[col].skew()
        if abs(skew) > 1.5:
            out.append(f"'{col}' is highly skewed (skew={skew:.2f}); consider a log transform.")
    return out


# ---------- 2. Cleaning ----------
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log, df = [], df.copy()
    n = len(df)
    df = df.drop_duplicates()
    if len(df) < n:
        log.append(f"Removed {n - len(df)} duplicate rows.")
    for col in df.columns:
        miss = df[col].isna().sum()
        if miss == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            log.append(f"Filled {miss} missing in '{col}' with median.")
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
            log.append(f"Filled {miss} missing in '{col}' with mode.")
    return df, log or ["Nothing to clean."]


# ---------- 3. AutoML ----------
def detect_task(y: pd.Series) -> str:
    return "classification" if (not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10) else "regression"


def auto_ml(df: pd.DataFrame, target: str, random_state: int = 42) -> dict:
    df = df.dropna(subset=[target])
    X, y = df.drop(columns=[target]), df[target]
    task = detect_task(y)
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    if task == "classification":
        models = {"Logistic Regression": LogisticRegression(max_iter=1000),
                  "Random Forest": RandomForestClassifier(n_estimators=150, random_state=random_state),
                  "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)}
        scoring = "f1_weighted"
    else:
        models = {"Linear Regression": LinearRegression(),
                  "Random Forest": RandomForestRegressor(n_estimators=150, random_state=random_state),
                  "Gradient Boosting": GradientBoostingRegressor(random_state=random_state)}
        scoring = "r2"

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
        stratify=y if task == "classification" and y.value_counts().min() >= 2 else None)

    leaderboard = []
    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        cv = cross_val_score(pipe, X_tr, y_tr, cv=3, scoring=scoring)
        leaderboard.append({"Model": name, f"CV {scoring}": round(cv.mean(), 4)})
    lb = pd.DataFrame(leaderboard).sort_values(f"CV {scoring}", ascending=False).reset_index(drop=True)

    best_name = lb.loc[0, "Model"]
    best = Pipeline([("pre", pre), ("model", models[best_name])]).fit(X_tr, y_tr)
    pred = best.predict(X_te)
    if task == "classification":
        metrics = {"Accuracy": accuracy_score(y_te, pred), "F1 (weighted)": f1_score(y_te, pred, average="weighted")}
    else:
        metrics = {"R²": r2_score(y_te, pred), "MAE": mean_absolute_error(y_te, pred)}

    imp = permutation_importance(best, X_te, y_te, n_repeats=5, random_state=random_state)
    importance = (pd.DataFrame({"Feature": X.columns, "Importance": imp.importances_mean})
                  .sort_values("Importance", ascending=False).reset_index(drop=True))
    return {"task": task, "leaderboard": lb, "best_model": best_name,
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "importance": importance, "pipeline": best}


# ---------- 4. Clustering ----------
def cluster(df: pd.DataFrame, k: int | None = None, random_state: int = 42) -> dict:
    num = df.select_dtypes(include="number").dropna()
    if num.shape[1] < 2 or len(num) < 10:
        raise ValueError("Need at least 2 numeric columns and 10 rows.")
    Xs = StandardScaler().fit_transform(num)

    scores = {}
    if k is None:
        for kk in range(2, min(8, len(num) - 1)):
            labels = KMeans(kk, n_init=10, random_state=random_state).fit_predict(Xs)
            scores[kk] = silhouette_score(Xs, labels)
        k = max(scores, key=scores.get)

    labels = KMeans(k, n_init=10, random_state=random_state).fit_predict(Xs)
    coords = PCA(2, random_state=random_state).fit_transform(Xs)
    plot = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": labels.astype(str)})
    profile_df = num.assign(Cluster=labels).groupby("Cluster").mean().round(2)
    profile_df["Size"] = pd.Series(labels).value_counts().sort_index().values
    return {"k": k, "silhouette": round(float(silhouette_score(Xs, labels)), 3),
            "plot": plot, "profile": profile_df, "scores": scores}


# ---------- 5. Anomaly detection ----------
def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    X = SimpleImputer(strategy="median").fit_transform(num)
    iso = IsolationForest(contamination=contamination, random_state=random_state).fit(X)
    out = df.copy()
    out["anomaly_score"] = -iso.score_samples(X)          # higher = more anomalous
    out["is_anomaly"] = iso.predict(X) == -1
    return out.sort_values("anomaly_score", ascending=False)
