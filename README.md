# ML Model Deployment with FastAPI and Docker

A RandomForest classifier trained on the classic **Iris** dataset, served as a REST API with **FastAPI** and packaged with **Docker**.

## Project structure

```
ml-fastapi-docker/
├── main.py            # FastAPI app (GET /, POST /predict)
├── train.py           # trains and saves the model
├── model.joblib       # saved model
├── requirements.txt   # Python dependencies
├── Dockerfile         # container instructions
└── README.md
```

## Step 1. Train the model

```bash
pip install -r requirements.txt
python train.py
```

Output: `Test accuracy: ...` and the file `model.joblib`.

## Step 2–3. Run the API locally

```bash
uvicorn main:app --reload
```

- Root: http://127.0.0.1:8000/ → `{"message": "ML API is running"}`
- Swagger docs: http://127.0.0.1:8000/docs

Test the prediction endpoint:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Response:

```json
{"prediction": 0, "class_name": "setosa", "probability": 1.0}
```

More examples:

| Input (sl, sw, pl, pw) | Expected class |
|---|---|
| 5.1, 3.5, 1.4, 0.2 | setosa |
| 6.0, 2.7, 4.5, 1.5 | versicolor |
| 6.9, 3.1, 5.8, 2.3 | virginica |

## Step 4–5. Build the Docker image

```bash
docker build -t ml-fastapi-docker .
```

## Step 6. Run the container

```bash
docker run -d -p 8000:8000 --name ml-api ml-fastapi-docker
```

Then test again: http://localhost:8000/, http://localhost:8000/docs, and the `curl` request above.

Stop and remove:

```bash
docker stop ml-api && docker rm ml-api
```
