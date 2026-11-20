import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timezone
from backend.schemas import CustomerInput, ChurnReason, PredictionOutput
from pydantic import ValidationError
from backend.schemas import BatchPredictionResponse, RowWarning


BASE_DIR=Path(__file__).resolve().parent.parent
MODEL_PATH=BASE_DIR / "artifacts" / "churn_prediction_pipeline.pkl"
METRICS_PATH=BASE_DIR / "artifacts" / "metrics.json"

MODEL_VERSION="v1.0.0"

try:
    pipeline=joblib.load(MODEL_PATH)

except Exception as e:
    pipeline=None
    print(f"Error Loading Model Pipeline {e}")

try:
    with open(METRICS_PATH,"r") as f:
        model_metrics=json.load(f)
except Exception as e:
    model_metrics={}
    print(f"Error Loading Model Metrics {e}")


def get_risk_tier(probability:float)->str:
    if probability >= 0.70:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    return "Low"



FEATURE_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender",
    "Dependents",
    "Contract",
]




def get_local_reasons(df: pd.DataFrame, top_n: int = 3) -> list[ChurnReason]:
    """Calculates exact feature contributions using Logistic Regression weights."""
    preprocessor = pipeline[:-1]
    classifier = pipeline[-1]

    
    X_transformed = preprocessor.transform(df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    
    contributions = X_transformed[0] * coefficients

    reasons = []
    for raw_name, impact in zip(feature_names, contributions):
        # Clean scikit-learn prefixes like "num__tenure" -> "tenure"
        clean_name = raw_name.split("__")[-1]
        direction = "Increases Churn Risk" if impact > 0 else "Reduces Churn Risk"
        reasons.append(
            ChurnReason(
                feature=clean_name,
                impact=round(float(impact), 4),
                direction=direction,
            )
        )

    
    reasons.sort(key=lambda r: abs(r.impact), reverse=True)
    return reasons[:top_n]


def predict_single(customer: CustomerInput) -> PredictionOutput:
    """Scores a single validated customer and returns probability, risk tier, and top reasons."""
    if pipeline is None:
        raise RuntimeError("Model pipeline is not loaded.")

    
    input_dict = customer.model_dump()
    row_data = {col: [input_dict[col]] for col in FEATURE_COLS}
    df = pd.DataFrame(row_data)

    
    prob = float(pipeline.predict_proba(df)[0][1])
    pred = 1 if prob >= 0.5 else 0

    
    cust_id = customer.customer_id or f"CUST-{uuid.uuid4().hex[:6].upper()}"

    return PredictionOutput(
        customer_id=cust_id,
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_tier=get_risk_tier(prob),
        top_reasons=get_local_reasons(df, top_n=3),
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def predict_batch(raw_rows: list[dict]) -> BatchPredictionResponse:
    """Scores a list of raw customer dictionaries safely without crashing on bad/duplicate rows."""
    if pipeline is None:
        raise RuntimeError("Model pipeline is not loaded.")

    predictions = []
    warnings = []
    seen_ids = set()

    for idx, raw_row in enumerate(raw_rows, start=1):
        
        cleaned_row = {
            k: (None if pd.isna(v) or v == "" else v) for k, v in raw_row.items()
        }

        raw_id = cleaned_row.get("customer_id")
        str_id = str(raw_id).strip() if raw_id is not None else None
        cleaned_row["customer_id"] = str_id

        
        if str_id and str_id in seen_ids:
            warnings.append(
                RowWarning(
                    row_number=idx,
                    customer_id=str_id,
                    issue="Duplicate customer_id skipped.",
                )
            )
            continue

        
        try:
            validated_customer = CustomerInput(**cleaned_row)
        except ValidationError as e:
            # Extract a short, readable error summary from Pydantic
            first_err = e.errors()[0]
            field_name = first_err.get("loc", ["unknown"])[0]
            msg = first_err.get("msg", "Invalid value")
            warnings.append(
                RowWarning(
                    row_number=idx,
                    customer_id=str_id,
                    issue=f"Invalid '{field_name}': {msg}",
                )
            )
            continue

        
        feature_values = [getattr(validated_customer, col) for col in FEATURE_COLS]
        if all(val is None for val in feature_values):
            warnings.append(
                RowWarning(
                    row_number=idx,
                    customer_id=str_id,
                    issue="All 6 feature values are missing; row skipped.",
                )
            )
            continue

        
        result = predict_single(validated_customer)
        predictions.append(result)
        if str_id:
            seen_ids.add(str_id)

    high_risk_count = sum(1 for p in predictions if p.risk_tier == "High")

    return BatchPredictionResponse(
        total_rows_received=len(raw_rows),
        total_processed=len(predictions),
        skipped_count=len(warnings),
        high_risk_count=high_risk_count,
        predictions=predictions,
        warnings=warnings,
    )


def get_model_info() -> dict:
    """Returns saved metrics.json and global Logistic Regression feature weights for Page 6."""
    global_importance = []

    if pipeline is not None:
        preprocessor = pipeline[:-1]
        classifier = pipeline[-1]
        feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_[0]

        for raw_name, coef in zip(feature_names, coefficients):
            clean_name = raw_name.split("__")[-1]
            global_importance.append(
                {
                    "feature": clean_name,
                    "coefficient": round(float(coef), 4),
                    "direction": "Increases Churn Risk" if coef > 0 else "Reduces Churn Risk",
                }
            )
        global_importance.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

    return {
        "model_version": MODEL_VERSION,
        "algorithm": "LogisticRegression (C=10.0, class_weight='balanced')",
        "metrics": model_metrics,
        "global_feature_importance": global_importance,
    }

