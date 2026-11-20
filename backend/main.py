import io
from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    CustomerInput,
    PredictionOutput,
    BatchPredictionResponse,
    HealthResponse,
)
from backend.predictor import (
    MODEL_VERSION,
    pipeline,
    predict_single,
    predict_batch,
    get_model_info,
)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time and batch customer churn scoring with exact feature explainability.",
    version=MODEL_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    is_loaded = pipeline is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_version=MODEL_VERSION,
    )


@app.get("/model-info")
def model_info():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    return get_model_info()


@app.post("/predict", response_model=PredictionOutput)
def predict_customer(customer: CustomerInput):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")

    try:
        return predict_single(customer)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_customers_batch(rows: List[Dict[str, Any]]):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")

    if not rows:
        raise HTTPException(status_code=400, detail="Batch payload cannot be empty.")

    try:
        return predict_batch(rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/predict/csv", response_model=BatchPredictionResponse)
async def predict_customers_csv(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    raw_rows = df.to_dict(orient="records")
    return predict_batch(raw_rows)
