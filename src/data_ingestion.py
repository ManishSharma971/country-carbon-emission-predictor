from pathlib import Path
from io import BytesIO
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
COLS = ["country", "year", "co2", "population", "gdp"]

def ingest_data(url=URL, output_path=None):
    """Download and clean source data into a compact CSV for training."""
    output_path = Path(output_path) if output_path else DATA_DIR / "cleaned_co2.csv"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    df = pd.read_csv(BytesIO(response.content), usecols=lambda c: c in COLS, low_memory=False)
    missing = set(COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    for col in ["year", "co2", "population", "gdp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=COLS)
    df = df[(df.co2 >= 0) & (df.population > 0) & (df.gdp > 0) & (df.year >= 1990)]
    df = df.sort_values(["country", "year"]).drop_duplicates(["country", "year"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")
    return df

if __name__ == "__main__":
    ingest_data()
