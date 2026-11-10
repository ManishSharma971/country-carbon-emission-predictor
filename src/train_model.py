from pathlib import Path
import json, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned_co2.csv"
ART = ROOT / "artifacts"
CHARTS = ROOT / "visualizations"
NUM = ["year", "population", "gdp"]
CAT = ["country"]
TARGET = "co2"

def make_pipeline(estimator):
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
        ("country", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("encode", OneHotEncoder(handle_unknown="ignore"))]), CAT)
    ])
    return Pipeline([("prep", prep), ("model", estimator)])

def train_and_evaluate(data_path=DATA):
    if not Path(data_path).exists():
        raise FileNotFoundError("Run `python -m src.data_ingestion` first.")
    df = pd.read_csv(data_path).dropna(subset=NUM + CAT + [TARGET])
    if len(df) < 100:
        raise ValueError("At least 100 clean rows are required.")
    X, y = df[NUM + CAT], df[TARGET].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    estimators = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
    }
    fitted, scores = {}, {}
    for name, est in estimators.items():
        model = make_pipeline(est).fit(Xtr, ytr)
        pred = np.maximum(0, model.predict(Xte))
        scores[name] = {"rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                        "mae": float(mean_absolute_error(yte, pred)),
                        "r2": float(r2_score(yte, pred))}
        fitted[name] = model
        print(f"{name}: RMSE={scores[name]['rmse']:.3f}, MAE={scores[name]['mae']:.3f}, R2={scores[name]['r2']:.3f}")
    best = min(["linear_regression", "random_forest"], key=lambda n: scores[n]["rmse"])
    ART.mkdir(exist_ok=True); CHARTS.mkdir(exist_ok=True)
    joblib.dump(fitted[best], ART / "best_model.joblib")
    metrics = {"selected_model": best, "scores": scores, "test_rows": len(Xte),
               "features": NUM + CAT, "target": TARGET}
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plt.figure(figsize=(9,5)); df.groupby("year")["co2"].sum().plot()
    plt.title("Recorded CO2 by year (sum of available country rows)"); plt.xlabel("Year"); plt.ylabel("CO2")
    plt.tight_layout(); plt.savefig(CHARTS / "eda_emissions_by_year.png", dpi=130); plt.close()
    plt.figure(figsize=(9,5)); df["co2"].clip(upper=df.co2.quantile(.99)).hist(bins=40)
    plt.title("CO2 distribution (clipped at 99th percentile)"); plt.tight_layout()
    plt.savefig(CHARTS / "eda_co2_distribution.png", dpi=130); plt.close()
    labels = ["Mean baseline", best.replace("_", " ").title()]
    values = [scores["mean_baseline"]["rmse"], scores[best]["rmse"]]
    plt.figure(figsize=(8,5)); bars=plt.bar(labels, values); plt.ylabel("Test RMSE")
    plt.title("Selected model vs baseline (lower is better)")
    for b,v in zip(bars,values): plt.text(b.get_x()+b.get_width()/2,b.get_height(),f"{v:.2f}",ha="center",va="bottom")
    plt.tight_layout(); plt.savefig(CHARTS / "model_vs_baseline.png", dpi=130); plt.close()
    pred = np.maximum(0, fitted[best].predict(Xte)); n=min(2000,len(yte)); ix=np.random.default_rng(42).choice(len(yte),n,replace=False)
    plt.figure(figsize=(6,6)); plt.scatter(yte.iloc[ix],pred[ix],alpha=.3,s=10)
    plt.xlabel("Actual CO2"); plt.ylabel("Predicted CO2"); plt.title("Actual vs predicted")
    plt.tight_layout(); plt.savefig(CHARTS / "actual_vs_predicted.png", dpi=130); plt.close()
    print(f"Selected model: {best}. Artifacts and charts saved.")
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
