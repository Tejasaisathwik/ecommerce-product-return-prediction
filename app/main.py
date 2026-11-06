"""FastAPI application for E-Commerce Product Return Prediction."""

from __future__ import annotations
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    OrderReturnInput,
    PredictionResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ecommerce-return-api")

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(PROJECT_ROOT / "models" / "best_model.joblib"),
)
METRICS_PATH = os.getenv(
    "METRICS_PATH",
    str(PROJECT_ROOT / "models" / "metrics.json"),
)

# In-memory storage for loaded model artifact
model_store = {
    "pipeline": None,
    "model_name": "Unknown",
    "metrics": {},
}


def load_artifacts():
    """Load model artifact and evaluation metrics."""
    logger.info("Loading model from: %s", MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        try:
            model_store["pipeline"] = joblib.load(MODEL_PATH)
            logger.info("Successfully loaded ML pipeline artifact.")
        except Exception as exc:
            logger.error("Failed to load model pipeline: %s", exc)
            model_store["pipeline"] = None
    else:
        logger.warning("Model file not found at: %s", MODEL_PATH)

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                model_store["metrics"] = json.load(f)
                model_store["model_name"] = model_store["metrics"].get("best_model", "BestModel")
        except Exception as exc:
            logger.warning("Failed to load metrics JSON: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    load_artifacts()
    yield
    logger.info("Shutting down E-Commerce Return Prediction API service.")


app = FastAPI(
    title="E-Commerce Product Return Prediction API",
    description=(
        "Production-grade Machine Learning inference service to predict whether "
        "a purchased product is likely to be returned based on customer behavior, "
        "order logistics, product category, and package characteristics."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    """Root entry point with service metadata and documentation links."""
    return {
        "service": "E-Commerce Product Return Prediction API",
        "version": "1.2.0",
        "status": "online",
        "model_loaded": model_store["pipeline"] is not None,
        "model_name": model_store["model_name"],
        "ci_cd_pipeline": "Automated via GitHub Actions & GHCR",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "metrics": "/metrics",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["System"],
)
def health():
    """Health check endpoint to verify service and model readiness."""
    pipeline_loaded = model_store["pipeline"] is not None
    return HealthResponse(
        status="healthy" if pipeline_loaded else "degraded",
        model_loaded=pipeline_loaded,
        model_name=model_store["model_name"],
        app_version="1.2.0",
        ci_cd_pipeline="Automated via GitHub Actions & GHCR",
        environment="production",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def predict(order: OrderReturnInput):
    """Predict whether a single order will be returned or kept.

    Computes engineered features (delivery_delay, customer_return_frequency,
    discount_amount, effective_price, is_delayed) and runs the complete
    validated scikit-learn pipeline.
    """
    pipeline = model_store["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model artifact is not loaded.",
        )

    # Convert Pydantic model to DataFrame for inference
    input_dict = order.model_dump()
    df_input = pd.DataFrame([input_dict])

    try:
        raw_pred = pipeline.predict(df_input)[0]
        probabilities = pipeline.predict_proba(df_input)[0]
        prob_return = float(probabilities[1])

        label = "Returned" if raw_pred == 1 else "Not Returned"
        is_returned = bool(raw_pred == 1)

        # Categorize risk level based on return probability
        if prob_return >= 0.65:
            risk_tier = "High Risk"
        elif prob_return >= 0.40:
            risk_tier = "Moderate Risk"
        else:
            risk_tier = "Low Risk"

        # Extract engineered features for transparent explanation
        delay = float(order.Actual_Delivery_Days - order.Expected_Delivery_Days)
        freq = float(order.Previous_Returns / max(order.Previous_Orders, 1))
        disc_amt = float(order.Product_Price_INR * (order.Discount_Percent / 100.0))

        return PredictionResponse(
            prediction=label,
            is_returned=is_returned,
            return_probability=round(prob_return, 4),
            risk_level=risk_tier,
            model_name=model_store["model_name"],
            engineered_features={
                "delivery_delay_days": delay,
                "is_delayed": delay > 0,
                "customer_return_frequency": round(freq, 4),
                "discount_amount_inr": round(disc_amt, 2),
                "effective_price_inr": round(order.Product_Price_INR - disc_amt, 2),
            },
        )
    except Exception as exc:
        logger.error("Error during model inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {str(exc)}",
        )


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def batch_predict(batch: BatchPredictionRequest):
    """Batch inference endpoint to process multiple orders simultaneously."""
    pipeline = model_store["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model artifact is not loaded.",
        )

    if not batch.orders:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list cannot be empty.",
        )

    orders_data = [o.model_dump() for o in batch.orders]
    df_batch = pd.DataFrame(orders_data)

    try:
        raw_preds = pipeline.predict(df_batch)
        probabilities = pipeline.predict_proba(df_batch)[:, 1]

        results = []
        for i, order in enumerate(batch.orders):
            is_ret = bool(raw_preds[i] == 1)
            label = "Returned" if is_ret else "Not Returned"
            prob = float(probabilities[i])

            if prob >= 0.65:
                risk_tier = "High Risk"
            elif prob >= 0.40:
                risk_tier = "Moderate Risk"
            else:
                risk_tier = "Low Risk"

            delay = float(order.Actual_Delivery_Days - order.Expected_Delivery_Days)
            freq = float(order.Previous_Returns / max(order.Previous_Orders, 1))
            disc_amt = float(order.Product_Price_INR * (order.Discount_Percent / 100.0))

            results.append(
                PredictionResponse(
                    prediction=label,
                    is_returned=is_ret,
                    return_probability=round(prob, 4),
                    risk_level=risk_tier,
                    model_name=model_store["model_name"],
                    engineered_features={
                        "delivery_delay_days": delay,
                        "is_delayed": delay > 0,
                        "customer_return_frequency": round(freq, 4),
                        "discount_amount_inr": round(disc_amt, 2),
                        "effective_price_inr": round(order.Product_Price_INR - disc_amt, 2),
                    },
                )
            )

        return BatchPredictionResponse(
            total_records=len(results),
            predictions=results,
        )
    except Exception as exc:
        logger.error("Error during batch inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(exc)}",
        )


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Return offline model validation metrics and confusion matrix."""
    if not model_store["metrics"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metrics report not found.",
        )
    return model_store["metrics"]
