"""FastAPI service that serves predictions from the Iris model.

SIS-3: the model is loaded from the MLflow Model Registry
(models:/iris-classifier@champion). If MLflow is not reachable,
the API falls back to the local model.joblib file (as in Task 6).
"""
import logging
import os
import time
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "iris-classifier")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

log = logging.getLogger("uvicorn.error")

app = FastAPI(title="Iris ML API", description="Predicts the Iris flower species", version="2.0.0")


def load_model(attempts: int = 3):
    """Load the champion model from MLflow (with retries); fall back to model.joblib."""
    for attempt in range(1, attempts + 1):
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(TRACKING_URI)
            version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
            info = {"source": "mlflow-registry", "model_name": MODEL_NAME, "alias": MODEL_ALIAS,
                    "version": version.version, "run_id": version.run_id,
                    "tracking_uri": TRACKING_URI}
            log.info("Loaded %s v%s from MLflow", MODEL_NAME, version.version)
            return model, TARGET_NAMES, info
        except Exception as exc:
            log.warning("Attempt %d/%d to load model from MLflow failed: %s",
                        attempt, attempts, str(exc).splitlines()[0])
            if attempt < attempts:
                time.sleep(2 * attempt)

    # MLflow not available -> use the local file (same as Task 6)
    log.warning("MLflow model not available; using %s", MODEL_PATH.name)
    artifact = joblib.load(MODEL_PATH)
    info = {"source": "local-file", "model_name": artifact.get("model_name", MODEL_NAME),
            "version": artifact.get("model_version"), "run_id": artifact.get("run_id"),
            "file": MODEL_PATH.name}
    return artifact["model"], artifact["target_names"], info


model, target_names, model_info = load_model()


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1], description="Sepal length (cm)")
    sepal_width: float = Field(..., gt=0, examples=[3.5], description="Sepal width (cm)")
    petal_length: float = Field(..., gt=0, examples=[1.4], description="Petal length (cm)")
    petal_width: float = Field(..., gt=0, examples=[0.2], description="Petal width (cm)")


class Prediction(BaseModel):
    prediction: int
    class_name: str
    probability: float
    probabilities: dict[str, float]
    model_version: str | None = None


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.get("/model-info")
def get_model_info():
    """Which model (name / version / source) is currently served."""
    return model_info


@app.post("/reload")
def reload_model():
    """Reload the champion model (e.g. after a new version is registered)."""
    global model, target_names, model_info
    model, target_names, model_info = load_model()
    return model_info


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures):
    x = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    proba = model.predict_proba(x)[0]
    pred = int(proba.argmax())
    return Prediction(
        prediction=pred,
        class_name=target_names[pred],
        probability=round(float(proba[pred]), 4),
        probabilities={name: round(float(p), 4) for name, p in zip(target_names, proba)},
        model_version=None if model_info.get("version") is None else str(model_info["version"]),
    )
