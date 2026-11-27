# Python base image
FROM python:3.12-slim

# Working directory inside the container
WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (code + model.joblib)
COPY . .

# Train once at build time (local sqlite MLflow) so the image also works standalone,
# with model.joblib matching the installed scikit-learn version.
# In docker-compose the "trainer" service retrains against the MLflow server.
RUN python train.py && rm -rf mlflow.db mlruns

EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
