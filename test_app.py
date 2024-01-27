from fastapi.testclient import TestClient
import pytest
from app import app

client = TestClient(app)  # Initialize a test client for our FastAPI app

def test_get_form():
    """
    Confirming the proper functioning of the form
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers['content-type'] == "text/html; charset=utf-8"

def test_predict():
    """
    Verifying that the prediction endpoint responds correctly to valid input
    """
    test_query = {"text": "text for prediction"}
    response = client.post("/predict", json=test_query)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_validation():
    """
    Testing the prediction endpoint's response to empty input
    """
    test_query = {"text": ""}
    response = client.post("/predict", json=test_query)
    assert response.status_code == 400  # Expecting Bad Request status for empty input

@pytest.fixture(scope="module")
def test_client():
    with TestClient(app) as client:
        yield client
