"""Streamlit frontend for the Iris ML API (SIS-3).

The user moves sliders, the app sends the features to the FastAPI
/predict endpoint and shows the predicted species, probabilities and
which MLflow model version answered.
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5050")

EXAMPLES = {
    "Custom": None,
    "Setosa example": (5.1, 3.5, 1.4, 0.2),
    "Versicolor example": (6.0, 2.7, 4.5, 1.5),
    "Virginica example": (6.9, 3.1, 5.8, 2.3),
}
EMOJI = {"setosa": "🌸", "versicolor": "🌼", "virginica": "🌺"}

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")
st.title("🌸 Iris Species Classifier")
st.caption("Streamlit → FastAPI → model from the MLflow Model Registry")

# --- Sidebar: API status and model info -----------------------------------
with st.sidebar:
    st.header("Service status")
    st.write(f"API: `{API_URL}`")
    try:
        info = requests.get(f"{API_URL}/model-info", timeout=5).json()
        st.success("API is running")
        st.json(info)
    except requests.RequestException:
        info = None
        st.error("API is not reachable. Start it with `uvicorn main:app`.")
    if st.button("Reload model from registry"):
        try:
            st.json(requests.post(f"{API_URL}/reload", timeout=30).json())
        except requests.RequestException as exc:
            st.error(f"Reload failed: {exc}")
    st.markdown(f"[Open MLflow UI]({MLFLOW_UI_URL})")

# --- Input form ------------------------------------------------------------
choice = st.selectbox("Load an example", list(EXAMPLES))
default = EXAMPLES[choice] or (5.8, 3.0, 4.3, 1.3)

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, default[0], 0.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, default[1], 0.1)
with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, default[2], 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, default[3], 0.1)

payload = {
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width,
}

if st.button("Predict", type="primary", use_container_width=True):
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
    except requests.RequestException as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        name = result["class_name"]
        st.success(f"{EMOJI.get(name, '')} Predicted species: **{name}** "
                   f"(confidence {result['probability']:.0%})")
        probs = pd.DataFrame({"species": list(result["probabilities"]),
                              "probability": list(result["probabilities"].values())})
        st.bar_chart(probs, x="species", y="probability")
        st.caption(f"Answered by model version: {result.get('model_version') or 'local file'}")
        with st.expander("Raw API request / response"):
            st.json({"request": payload, "response": result})
