# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import pandas as pd
from typing import Dict, List, Any

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# ---------- Paths (Docker-safe) ----------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


# ---------- Schemas ----------
class Customer(BaseModel):
    # Keeping your current payload style: {"data": {...}}
    data: Dict[str, Any] = Field(
        ...,
        example={
            "tenure": 12,
            "MonthlyCharges": 70.35,
            "TotalCharges": 850.0,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "TechSupport": "No",
            "OnlineBackup": "No",
            "StreamingTV": "Yes",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "Partner": "No",
            "gender": "Male",
            "DeviceProtection": "No",
            "PaperlessBilling": "Yes",
            "Dependents": "No",
            "StreamingMovies": "Yes",
        },
    )


class BatchRequest(BaseModel):
    records: List[Dict[str, Any]]


# ---------- Load artifacts on startup ----------
@app.on_event("startup")
def load_artifacts() -> None:
    try:
        app.state.model = joblib.load(ARTIFACTS_DIR / "churn_pipeline.joblib")
        app.state.threshold = float((ARTIFACTS_DIR / "threshold.txt").read_text().strip())
        app.state.model_version = "churn_pipeline.joblib"  # simple version tag
    except Exception as e:
        # If this fails, app should not start silently
        raise RuntimeError(f"Failed to load artifacts from {ARTIFACTS_DIR}: {e}") from e


# ---------- Health & Home ----------
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.get("/health")
def health():
    # Simple health check used by deployments/load balancers
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "model"),
        "threshold_loaded": hasattr(app.state, "threshold"),
        "model_version": getattr(app.state, "model_version", None),
    }


# ---------- Helpers ----------
def _predict_proba_df(df: pd.DataFrame) -> float:
    model = app.state.model
    proba = model.predict_proba(df)[:, 1]
    return proba


# ---------- Endpoints ----------
@app.post("/predict")
def predict(customer: Customer):
    try:
        df = pd.DataFrame([customer.data])

        probs = _predict_proba_df(df)
        proba = float(probs[0])

        threshold = float(app.state.threshold)
        pred = int(proba >= threshold)

        return {
            "churn_probability": proba,
            "churn_prediction": pred,
            "threshold_used": threshold,
            "model_version": getattr(app.state, "model_version", None),
        }

    except Exception as e:
        # Return readable error instead of a 500
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    try:
        df = pd.DataFrame(req.records)

        probs = _predict_proba_df(df)
        threshold = float(app.state.threshold)
        preds = (probs >= threshold).astype(int)

        return {
            "threshold_used": threshold,
            "model_version": getattr(app.state, "model_version", None),
            "results": [
                {"churn_probability": float(p), "churn_prediction": int(y)}
                for p, y in zip(probs, preds)
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nbathula575.github.io",  # your GitHub Pages domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)