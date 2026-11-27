"""Train Iris classifiers with MLflow experiment tracking and Model Registry.

What this script does (SIS-3):
1. Creates / reuses an MLflow experiment ("iris-classification").
2. Trains several candidate models; for every run it logs
   - parameters (hyperparameters),
   - metrics (accuracy, F1, precision, recall, CV accuracy),
   - artifacts (model.joblib, confusion matrix, classification report, MLflow model).
3. Registers every model in the MLflow Model Registry under one name
   ("iris-classifier"), so each run becomes a new version.
4. Marks the best version with the alias "champion" and tags it.
5. Saves the best model to model.joblib (fallback for the API, as in Task 6).

Tracking server is taken from MLFLOW_TRACKING_URI (default: local sqlite:///mlflow.db).
"""
import json
import os
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).parent / "model.joblib"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classification")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "iris-classifier")
CHAMPION_ALIAS = "champion"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Candidate models: (run name, model type, hyperparameters)
CANDIDATES = [
    ("rf_100_trees", "RandomForest", {"n_estimators": 100, "max_depth": None}),
    ("rf_50_trees_depth3", "RandomForest", {"n_estimators": 50, "max_depth": 3}),
    ("logreg_C1", "LogisticRegression", {"C": 1.0, "max_iter": 500}),
]


def build_model(model_type: str, params: dict):
    if model_type == "RandomForest":
        return RandomForestClassifier(random_state=RANDOM_STATE, **params)
    if model_type == "LogisticRegression":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_STATE, **params))
    raise ValueError(f"Unknown model type: {model_type}")


def train_one(run_name, model_type, params, data, target_names):
    X_train, X_test, y_train, y_test = data
    with mlflow.start_run(run_name=run_name) as run:
        model = build_model(model_type, params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 1) Parameters
        mlflow.log_params({"model_type": model_type, **params,
                           "test_size": TEST_SIZE, "random_state": RANDOM_STATE})

        # 2) Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "cv_accuracy_mean": cross_val_score(build_model(model_type, params),
                                                X_train, y_train, cv=5).mean(),
        }
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"dataset": "iris", "project": "SIS-3", "framework": "scikit-learn"})

        # 3) Artifacts: saved model file, confusion matrix, classification report
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            joblib.dump({"model": model, "target_names": target_names}, tmp / "model.joblib")

            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=target_names,
                                                    ax=ax, colorbar=False)
            ax.set_title(run_name)
            fig.tight_layout()
            fig.savefig(tmp / "confusion_matrix.png")
            plt.close(fig)

            (tmp / "classification_report.json").write_text(json.dumps(
                classification_report(y_test, y_pred, target_names=target_names, output_dict=True),
                indent=2))

            mlflow.log_artifacts(str(tmp), artifact_path="extra")

        # 4) MLflow model + registration in the Model Registry (creates a new version)
        signature = infer_signature(X_test, model.predict(X_test))
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_test[:2],
            # Model is created by us -> trust sklearn's tree storage type (needed for RandomForest)
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            # explicit requirements -> MLflow skips slow automatic dependency inference
            pip_requirements=[f"scikit-learn=={sklearn.__version__}", f"mlflow=={mlflow.__version__}"],
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"[{run_name}] accuracy={metrics['accuracy']:.3f} f1={metrics['f1_macro']:.3f} "
              f"-> registered {REGISTERED_MODEL_NAME} v{model_info.registered_model_version}")
        return {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "model": model,
            "metrics": metrics,
            "version": model_info.registered_model_version,
        }


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {TRACKING_URI} | experiment: {EXPERIMENT_NAME}")

    iris = load_iris()
    target_names = [str(n) for n in iris.target_names]
    data = train_test_split(iris.data, iris.target, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=iris.target)

    results = [train_one(name, mtype, params, data, target_names)
               for name, mtype, params in CANDIDATES]

    # Best model = highest F1, then highest CV accuracy as tie-breaker
    best = max(results, key=lambda r: (r["metrics"]["f1_macro"], r["metrics"]["cv_accuracy_mean"]))

    client = MlflowClient()
    client.update_registered_model(
        REGISTERED_MODEL_NAME, description="Iris species classifier (SIS-3, FastAPI + Streamlit)")
    for r in results:
        client.update_model_version(
            REGISTERED_MODEL_NAME, r["version"],
            description=f"{r['run_name']}: accuracy={r['metrics']['accuracy']:.3f}, "
                        f"f1={r['metrics']['f1_macro']:.3f}")
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, best["version"])
    client.set_model_version_tag(REGISTERED_MODEL_NAME, best["version"], "stage", "production")

    print(f"Best model: {best['run_name']} -> {REGISTERED_MODEL_NAME} "
          f"v{best['version']} (alias '{CHAMPION_ALIAS}')")

    # Local fallback file for the API (same format as in Task 6)
    joblib.dump({"model": best["model"], "target_names": target_names,
                 "model_name": REGISTERED_MODEL_NAME, "model_version": str(best["version"]),
                 "run_id": best["run_id"]}, MODEL_PATH)
    print(f"Best model also saved to {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
