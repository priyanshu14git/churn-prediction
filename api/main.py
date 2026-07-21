"""
api/main.py
-----------
FastAPI backend serving churn predictions from the trained LightGBM model.

Usage (from project root, with venv activated):
    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs to test it interactively (Swagger UI).
"""

import joblib
import pandas as pd
import shap
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/final_model.pkl")

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model once at startup, not per-request (much faster) ---
_saved = joblib.load(MODEL_PATH)
model = _saved["model"]
feature_columns = _saved["feature_columns"]
explainer = shap.TreeExplainer(model)


class CustomerInput(BaseModel):
    """
    Raw customer fields, matching the original Telco dataset columns
    (before one-hot encoding - the API handles encoding internally).
    """
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0, description="0 = No, 1 = Yes")
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.35)
    TotalCharges: float = Field(..., example=845.5)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: str
    top_factors: dict


def encode_customer(customer: CustomerInput) -> pd.DataFrame:
    """
    Convert raw customer input into the same one-hot encoded format
    the model was trained on, using the saved feature_columns as the
    source of truth for column names/order.
    """
    raw_df = pd.DataFrame([customer.model_dump()])
    encoded = pd.get_dummies(raw_df)

    # Add any missing columns (categories not present in this single
    # customer's input) as 0, and drop any extras, so columns match
    # the training data exactly.
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    return encoded


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running. Visit /docs to test it."}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    try:
        X = encode_customer(customer)

        proba = model.predict_proba(X)[0][1]
        prediction = "Yes" if proba >= 0.5 else "No"

        # SHAP explanation for this single prediction
        shap_values = explainer.shap_values(X)
        impact = pd.Series(shap_values[0], index=X.columns).sort_values(
            key=abs, ascending=False
        )
        top_factors = impact.head(5).round(4).to_dict()

        return PredictionResponse(
            churn_probability=round(float(proba), 4),
            churn_prediction=prediction,
            top_factors=top_factors,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))