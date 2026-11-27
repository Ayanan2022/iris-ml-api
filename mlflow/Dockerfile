# MLflow tracking server + Model Registry (sqlite backend, artifacts in a volume)
FROM python:3.12-slim
RUN pip install --no-cache-dir mlflow==3.18.0
WORKDIR /mlflow
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/data/mlflow.db", \
     "--artifacts-destination", "/mlflow/data/artifacts", "--serve-artifacts", \
     "--allowed-hosts", "mlflow,mlflow:5000,localhost,localhost:*,127.0.0.1,127.0.0.1:*", \
     "--uvicorn-opts", "--timeout-keep-alive 300"]
