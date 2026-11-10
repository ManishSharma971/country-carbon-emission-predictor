import pytest
from src.app import app

@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and "status" in r.get_json()

def test_missing_prediction_fields(client):
    assert client.post("/predict", json={}).status_code == 400

def test_invalid_year(client):
    assert client.post("/predict", json={"country":"India","year":"bad"}).status_code == 400

def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200 and "requests" in r.get_json()


def test_country_prediction(client):
    response = client.post(
        "/predict",
        json={"country": "India", "year": 2025}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["country"] == "India"
    assert data["year"] == 2025
    assert "predicted_co2" in data


def test_all_country_prediction(client):
    response = client.post(
        "/predict",
        json={"country": "all", "year": 2025}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["country"] == "all"
    assert data["year"] == 2025
    assert data["countries_included"] > 0
    assert "predicted_total_co2" in data

