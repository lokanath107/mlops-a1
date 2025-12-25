from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_prediction_endpoint():
    """
    Test /predict API endpoint
    """

    payload = {
        "age": 55,
        "trestbps": 140,
        "chol": 240,
        "thalach": 150,
        "oldpeak": 1.0,
        "ca": 0,
        "sex": 1,
        "cp": 2,
        "fbs": 0,
        "restecg": 1,
        "exang": 0,
        "slope": 2,
        "thal": 2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in [0, 1]
