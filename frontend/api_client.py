import os
import requests

# Uses environment variable if set (for cloud deployment), otherwise defaults to local FastAPI
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def check_health() -> dict:
    """Checks if the FastAPI backend is reachable and the ML model is loaded."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "offline", "model_loaded": False, "error": str(e)}


def predict_single(payload: dict) -> dict:
    """Sends a single customer's features to POST /predict and returns the prediction."""
    response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def predict_csv(uploaded_file) -> dict:
    """Sends an uploaded Streamlit CSV file to POST /predict/csv for batch scoring."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
    response = requests.post(f"{API_BASE_URL}/predict/csv", files=files, timeout=30)
    response.raise_for_status()
    return response.json()


def get_model_info() -> dict:
    """Fetches model metrics and global feature coefficients from GET /model-info."""
    response = requests.get(f"{API_BASE_URL}/model-info", timeout=10)
    response.raise_for_status()
    return response.json()
