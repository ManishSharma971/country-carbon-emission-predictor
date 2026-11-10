from pathlib import Path
import time
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from src.data_ingestion import DATA_DIR
from src.monitoring import configure_logger, record_request, get_metrics

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"
DATA_PATH = DATA_DIR / "cleaned_co2.csv"
app = Flask(__name__)
logger = configure_logger()

def get_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python -m src.data_ingestion` first.")
    df = pd.read_csv(DATA_PATH)
    return df.dropna(subset=["country", "year", "population", "gdp", "co2"])

def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run `python -m src.train_model` first.")
    return joblib.load(MODEL_PATH)

@app.before_request
def start_timer():
    request._started_at = time.perf_counter()

@app.after_request
def log_response(response):
    latency = (time.perf_counter() - getattr(request, "_started_at", time.perf_counter())) * 1000
    record_request(latency, response.status_code >= 400)
    logger.info("method=%s path=%s status=%s latency_ms=%.2f", request.method, request.path, response.status_code, latency)
    return response

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_available": MODEL_PATH.exists(), "data_available": DATA_PATH.exists()})

@app.get("/metrics")
def metrics():
    return jsonify(get_metrics())

@app.get("/countries")
def countries():
    try:
        return jsonify(sorted(get_data()["country"].unique().tolist()))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

@app.post("/predict")
def predict():
    body = request.get_json(silent=True) or {}
    country = str(body.get("country", "")).strip()
    try: year = int(body.get("year", 0))
    except (ValueError, TypeError): return jsonify({"error": "year must be an integer"}), 400
    if not country or not 1800 <= year <= 2200:
        return jsonify({"error": "Provide country and a year between 1800 and 2200"}), 400
    try:
        df, model = get_data(), get_model()
        latest = df.sort_values("year").groupby("country", as_index=False).tail(1)
        if country.lower() == "all":
            rows = latest[["country", "population", "gdp"]].copy()
            rows["year"] = year
            rows = rows[["year", "population", "gdp", "country"]]
            total = float(pd.Series(model.predict(rows)).clip(lower=0).sum())
            return jsonify({"country": "all", "year": year, "predicted_total_co2": total,
                            "countries_included": len(rows),
                            "note": "Illustrative aggregate; latest known population/GDP held constant."})
        match = latest[latest.country.str.casefold() == country.casefold()]
        if match.empty: return jsonify({"error": f"Country not found: {country}; try GET /countries"}), 404
        row = match.iloc[0]
        try:
            population = float(body.get("population", row.population))
            gdp = float(body.get("gdp", row.gdp))
        except (ValueError, TypeError): return jsonify({"error": "population and gdp must be numeric"}), 400
        if population <= 0 or gdp <= 0: return jsonify({"error": "population and gdp must be positive"}), 400
        features = pd.DataFrame([{"year": year, "population": population, "gdp": gdp, "country": row.country}])
        value = max(0.0, float(model.predict(features)[0]))
        return jsonify({"country": row.country, "year": year, "predicted_co2": value,
                        "population_used": population, "gdp_used": gdp,
                        "note": "Future predictions hold latest known population/GDP constant unless overridden."})
    except FileNotFoundError as exc: return jsonify({"error": str(exc)}), 503
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed; check server log."}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
