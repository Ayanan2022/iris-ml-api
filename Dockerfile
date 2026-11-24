# Python base image
FROM python:3.12-slim

# Working directory inside the container
WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (code + model.joblib)
COPY . .

# Retrain the model inside the image so it always matches the installed scikit-learn version
RUN python train.py

EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
