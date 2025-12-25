import joblib
import pandas as pd
import os

def test_model_prediction():
    """
    Test whether the trained model loads and produces a valid prediction
    """

    # Load model
    model_path = "model/model.pkl"
    assert os.path.exists(model_path), "Model file not found"

    model = joblib.load(model_path)

    # Dummy input matching training features
    sample_input = pd.DataFrame([{
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
    }])

    prediction = model.predict(sample_input)

    # Assertions
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]
