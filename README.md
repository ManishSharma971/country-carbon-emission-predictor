
# Country-Level CO2 Emissions Prediction

A beginner-friendly machine learning capstone that predicts country-level CO2 emissions using historical data from Our World in Data.

## Features

- Automated data ingestion and cleaning
- Exploratory data analysis (EDA) visualizations
- Comparison of a mean baseline, Linear Regression, and Random Forest
- Flask API for country-specific and aggregate predictions
- Request count, error count, and latency monitoring
- File-based application logging
- Automated tests using pytest

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib
- Flask
- pytest
- joblib

## Project Setup (Windows)

Run these commands from the project root directory.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the Project

### 1. Download and prepare the dataset

```powershell
python -m src.data_ingestion
```

The cleaned dataset is saved to `data/cleaned_co2.csv`.

### 2. Train and compare models

```powershell
python -m src.train_model
```

The training script compares a mean baseline, Linear Regression, and Random Forest. It saves the selected model and evaluation metrics under `artifacts/` and charts under `visualizations/`.

### 3. Start the API

```powershell
python -m src.app
```

Open `http://127.0.0.1:5000/health` to check API health.

Keep the API terminal open while sending requests from another terminal.

## API Usage

The prediction endpoint uses HTTP POST and JSON input.

### Predict emissions for one country

PowerShell example:

```powershell
$body = @{
    country = "India"
    year = 2030
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

### Predict aggregate emissions for all countries

```powershell
$body = @{
    country = "all"
    year = 2030
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

### Available endpoints

- `GET /health` - checks API, data, and model availability
- `GET /countries` - lists available countries
- `GET /metrics` - returns request counts, errors, and latency statistics
- `POST /predict` - predicts emissions for a country or an aggregate

The prediction endpoint accepts years from 1800 through 2200. It returns HTTP 400 for invalid input.

## Model Evaluation

The current recorded random train/test split produced these results:

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Mean baseline | 561.82 | 286.26 | -0.039 |
| Linear Regression | 138.31 | 35.18 | 0.937 |
| Random Forest | 36.48 | 9.62 | 0.996 |

Lower RMSE and MAE are better; higher R2 is generally better.

These are results from the current test split. 

## Visualizations

Generated charts are stored in `visualizations/`:

- `eda_co2_distribution.png`
- `eda_emissions_by_year.png`
- `actual_vs_predicted.png`
- `model_vs_baseline.png`

## Automated Tests

Run the test suite from the project root:

```powershell
python -m pytest -q
```

The current suite contains nine passing tests covering API behavior, prediction responses, model pipeline behavior, request metrics, and test logging.

## Data Source and Limitations

Dataset: Our World in Data CO2 and Greenhouse Gas Emissions.

Source: https://github.com/owid/co2-data

Review the source dataset's current license and terms and retain appropriate attribution.

Future predictions hold the latest known population and GDP values constant unless overridden for a single-country prediction. The outputs are illustrative model estimates, not official forecasts.


