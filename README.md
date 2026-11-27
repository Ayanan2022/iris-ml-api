# SIS-3: Iris ML System — FastAPI + Streamlit + MLflow + Docker

Continuation of **Practical Task 6** (RandomForest on Iris → FastAPI → Docker).
In SIS-3 the project is extended into a full ML system:

- **Streamlit frontend** for user interaction;
- **MLflow experiment tracking** (parameters, metrics, artifacts);
- **MLflow Model Registry** (model name `iris-classifier`, versions, alias `champion`);
- the API now loads the model **from the registry**;
- everything runs together with **Docker Compose**.

## Architecture

```
 ┌──────────────┐  HTTP   ┌──────────────┐  models:/iris-classifier@champion  ┌──────────────────┐
 │  Streamlit   │ ──────► │   FastAPI    │ ─────────────────────────────────► │  MLflow server   │
 │  :8501       │ ◄────── │   :8000      │                                    │  tracking +      │
 └──────────────┘  JSON   └──────────────┘                                    │  model registry  │
                                                                              │  :5050           │
                          ┌──────────────┐  params, metrics, artifacts,       │                  │
                          │  train.py    │  register model version            │                  │
                          │  (trainer)   │ ─────────────────────────────────► │                  │
                          └──────────────┘                                    └──────────────────┘
```

1. `train.py` trains 3 candidate models, logs every run to the MLflow experiment
   `iris-classification` and registers each as a new version of `iris-classifier`.
2. The best version (highest F1) gets the alias **`champion`** and tag `stage=production`.
3. FastAPI loads `models:/iris-classifier@champion` (falls back to `model.joblib` if MLflow is down).
4. Streamlit sends user input to `POST /predict` and displays the result.

## Project structure

```
ml-fastapi-docker/
├── train.py              # training + MLflow tracking + Model Registry
├── main.py               # FastAPI: GET /, GET /model-info, POST /predict, POST /reload
├── model.joblib          # best model (local fallback)
├── requirements.txt      # API / training dependencies
├── Dockerfile            # image for API and trainer
├── docker-compose.yml    # mlflow + trainer + api + frontend
├── mlflow/Dockerfile     # MLflow tracking server image
├── frontend/
│   ├── app.py            # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── screenshots/          # Task 6 + SIS-3 screenshots
└── README.md
```

## What is logged to MLflow

| Type | Values |
|---|---|
| Parameters | `model_type`, `n_estimators`, `max_depth`, `C`, `max_iter`, `test_size`, `random_state` |
| Metrics | `accuracy`, `f1_macro`, `precision_macro`, `recall_macro`, `cv_accuracy_mean` |
| Artifacts | `extra/model.joblib` (saved model file), `extra/confusion_matrix.png`, `extra/classification_report.json`, `model/` (MLflow sklearn model with signature + input example) |
| Registry | model **`iris-classifier`**, versions 1, 2, 3 …, alias **`champion`** → best version |

Candidate models:

| Run | Model | Hyperparameters |
|---|---|---|
| `rf_100_trees` | RandomForest | n_estimators=100, max_depth=None |
| `rf_50_trees_depth3` | RandomForest | n_estimators=50, max_depth=3 |
| `logreg_C1` | StandardScaler + LogisticRegression | C=1.0, max_iter=500 |

## Option A — run everything with Docker Compose (recommended)

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit frontend | http://localhost:8501 |
| FastAPI (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5050 |

> Port 8000 must be free: stop the old Task 6 container first (`docker stop ml-api`),
> or choose other host ports: `API_PORT=8002 MLFLOW_PORT=5051 UI_PORT=8502 docker compose up --build`.
> MLflow uses host port 5050 because macOS AirPlay already occupies 5000.

The `trainer` container runs once, logs the experiment, registers the model and exits;
then the API starts and loads the `champion` version.

Retrain (creates new versions in the registry) and reload the API:

```bash
docker compose run --rm trainer
curl -X POST http://localhost:8000/reload
```

Stop: `docker compose down` (add `-v` to also delete MLflow data).

## Option B — run locally without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r frontend/requirements.txt

python train.py                                   # logs to sqlite:///mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050   # terminal 1
uvicorn main:app --port 8000                      # terminal 2
MLFLOW_UI_URL=http://127.0.0.1:5050 streamlit run frontend/app.py   # terminal 3
```

## API

```bash
curl http://127.0.0.1:8000/model-info
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

```json
{"prediction": 0, "class_name": "setosa", "probability": 1.0,
 "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}, "model_version": "1"}
```

| Input (sl, sw, pl, pw) | Expected class |
|---|---|
| 5.1, 3.5, 1.4, 0.2 | setosa |
| 6.0, 2.7, 4.5, 1.5 | versicolor |
| 6.9, 3.1, 5.8, 2.3 | virginica |

## Notes on implementation

- MLflow 3 stores sklearn models with **skops**; RandomForest needs the trusted type
  `sklearn.tree._tree.Tree`, so `train.py` passes `skops_trusted_types` (the model is created by us).
- The MLflow server runs with `--allowed-hosts` (so containers can reach it as `mlflow:5000`)
  and a longer keep-alive timeout.
- The API retries loading from the registry 3 times and only then falls back to `model.joblib`.

## Screenshots to take for the report

1. MLflow UI — experiment `iris-classification` with 3 runs and metric columns.
2. MLflow UI — one run: Parameters, Metrics, Artifacts (`model.joblib`, confusion matrix).
3. MLflow UI — Models → `iris-classifier` with versions and alias `champion`.
4. Streamlit — prediction result with probability chart.
5. Swagger `/docs` — `/model-info` showing `source: mlflow-registry`.
6. Docker Desktop / `docker compose ps` — all containers running.
