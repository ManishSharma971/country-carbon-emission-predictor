
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from src.train_model import make_pipeline


def test_model_pipeline_fits_and_predicts():
    X = pd.DataFrame({
        "year": [2000, 2001, 2002, 2003],
        "population": [100, 110, 120, 130],
        "gdp": [1000, 1100, 1200, 1300],
        "country": ["A", "A", "B", "B"]
    })
    y = [1.0, 2.0, 3.0, 4.0]

    model = make_pipeline(DummyRegressor()).fit(X, y)
    predictions = model.predict(X.iloc[:1])

    assert len(predictions) == 1
    assert np.isfinite(predictions).all()
