import pytest
import json
import os

# Set test env vars before importing app
os.environ["API_KEY"] = "test_key_123"
os.environ["PRODUCTION"] = "false"

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def auth_headers():
    return {"X-API-Key": "test_key_123", "Content-Type": "application/json"}


# --- Health check ---

def test_home_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_home_returns_running_status(client):
    response = client.get("/")
    data = json.loads(response.data)
    assert data["status"] == "running"


def test_home_lists_endpoints(client):
    response = client.get("/")
    data = json.loads(response.data)
    assert "endpoints" in data
    assert "POST /predict" in data["endpoints"]


# --- Auth ---

def test_predict_rejects_missing_api_key(client):
    response = client.post(
        "/predict",
        data=json.dumps({"text": "hello"}),
        content_type="application/json"
    )
    assert response.status_code == 401


def test_predict_rejects_wrong_api_key(client):
    response = client.post(
        "/predict",
        data=json.dumps({"text": "hello"}),
        headers={"X-API-Key": "wrong_key", "Content-Type": "application/json"}
    )
    assert response.status_code == 401


# --- Predict endpoint ---

def test_predict_positive_sentiment(client):
    response = client.post(
        "/predict",
        data=json.dumps({"text": "I absolutely love this product!"}),
        headers=auth_headers()
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["prediction"]["label"] == "POSITIVE"
    assert data["prediction"]["score"] > 0.9


def test_predict_negative_sentiment(client):
    response = client.post(
        "/predict",
        data=json.dumps({"text": "This is terrible and I hate it."}),
        headers=auth_headers()
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["prediction"]["label"] == "NEGATIVE"


def test_predict_returns_input_text(client):
    text = "This is a test sentence."
    response = client.post(
        "/predict",
        data=json.dumps({"text": text}),
        headers=auth_headers()
    )
    data = json.loads(response.data)
    assert data["input_text"] == text


def test_predict_missing_text_field(client):
    response = client.post(
        "/predict",
        data=json.dumps({"wrong_field": "hello"}),
        headers=auth_headers()
    )
    assert response.status_code == 400


def test_predict_empty_text(client):
    response = client.post(
        "/predict",
        data=json.dumps({"text": "   "}),
        headers=auth_headers()
    )
    assert response.status_code == 400


def test_predict_no_body(client):
    response = client.post(
        "/predict",
        headers=auth_headers()
    )
    assert response.status_code == 400


def test_predict_score_is_rounded(client):
    response = client.post(
        "/predict",
        data=json.dumps({"text": "Great!"}),
        headers=auth_headers()
    )
    data = json.loads(response.data)
    score = data["prediction"]["score"]
    # Score should be rounded to 4 decimal places
    assert round(score, 4) == score
