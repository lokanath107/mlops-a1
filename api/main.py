from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load trained pipeline
model = joblib.load("model/model.pkl")

FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal'
]

@app.post("/predict")
def predict(data: dict):
    # Convert JSON → DataFrame
    df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}
