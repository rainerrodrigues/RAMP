from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="RAMP Predictive Maintenance API")

model = joblib.load("models/model.pkl")


class SensorFeatures(BaseModel):
    mean: float
    std: float
    rms: float
    kurtosis: float
    skewness: float
    peak_to_peak: float
    max: float
    min: float
    dominant_frequency: float
    spectral_energy: float


@app.get("/")
def home():
    return {"message": "RAMP Predictive Maintenance API running"}


@app.post("/predict")
def predict(features: SensorFeatures):

    data = np.array([
        [
            features.mean,
            features.std,
            features.rms,
            features.kurtosis,
            features.skewness,
            features.peak_to_peak,
            features.max,
            features.min,
            features.dominant_frequency,
            features.spectral_energy
        ]
    ])

    prediction = model.predict(data)

    return {
        "prediction": prediction[0]
    }
