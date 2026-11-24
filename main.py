"""FastAPI service that serves predictions from the trained Iris model."""
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"

app = FastAPI(title="Iris ML API", description="Predicts the Iris flower species", version="1.0.0")

# Load the model once at startup
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
target_names = artifact["target_names"]


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1], description="Sepal length (cm)")
    sepal_width: float = Field(..., gt=0, examples=[3.5], description="Sepal width (cm)")
    petal_length: float = Field(..., gt=0, examples=[1.4], description="Petal length (cm)")
    petal_width: float = Field(..., gt=0, examples=[0.2], description="Petal width (cm)")


class Prediction(BaseModel):
    prediction: int
    class_name: str
    probability: float


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures):
    x = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    pred = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[0][pred])
    return Prediction(prediction=pred, class_name=target_names[pred], probability=round(proba, 4))
