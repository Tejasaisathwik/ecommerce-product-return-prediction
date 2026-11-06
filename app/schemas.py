"""Pydantic schemas for request and response validation in FastAPI."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class OrderReturnInput(BaseModel):
    """Input payload representing an e-commerce order."""

    Product_Category: str = Field(
        ...,
        description="Category of the product: Beauty, Fashion, Electronics, Home, Sports",
        examples=["Fashion"],
    )
    Product_Price_INR: float = Field(
        ...,
        description="Product retail price in Indian Rupees (INR)",
        gt=0,
        examples=[12825.0],
    )
    Discount_Percent: float = Field(
        ...,
        description="Discount percentage offered on the order (0 - 100)",
        ge=0.0,
        le=100.0,
        examples=[25.0],
    )
    Expected_Delivery_Days: int = Field(
        ...,
        description="Estimated delivery duration in days promised to customer",
        ge=1,
        examples=[6],
    )
    Actual_Delivery_Days: int = Field(
        ...,
        description="Actual number of days taken for the package to be delivered",
        ge=1,
        examples=[9],
    )
    Customer_Rating_PreOrder: float = Field(
        ...,
        description="Customer historical or pre-order satisfaction rating (1.0 to 5.0)",
        ge=1.0,
        le=5.0,
        examples=[3.8],
    )
    Previous_Returns: int = Field(
        ...,
        description="Count of past returned orders by this customer",
        ge=0,
        examples=[3],
    )
    Previous_Orders: int = Field(
        ...,
        description="Total count of lifetime previous orders placed by this customer",
        ge=0,
        examples=[12],
    )
    Payment_Method: str = Field(
        ...,
        description="Payment mode used: Card, COD, UPI, Wallet",
        examples=["COD"],
    )
    Damaged_Package_Flag: str = Field(
        ...,
        description="Whether package arrived damaged: Yes or No",
        examples=["No"],
    )
    Package_Size: str = Field(
        ...,
        description="Physical dimension tier: Small, Medium, Large",
        examples=["Medium"],
    )
    Promotion_Active: str = Field(
        ...,
        description="Whether promotion / campaign was active during purchase: Yes or No",
        examples=["Yes"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Product_Category": "Fashion",
                "Product_Price_INR": 12825.0,
                "Discount_Percent": 25.0,
                "Expected_Delivery_Days": 5,
                "Actual_Delivery_Days": 8,
                "Customer_Rating_PreOrder": 3.8,
                "Previous_Returns": 3,
                "Previous_Orders": 10,
                "Payment_Method": "COD",
                "Damaged_Package_Flag": "No",
                "Package_Size": "Medium",
                "Promotion_Active": "Yes",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Output prediction response for an individual product order."""

    prediction: str = Field(
        ...,
        description="Classification label: 'Returned' or 'Not Returned'",
        examples=["Returned"],
    )
    is_returned: bool = Field(
        ...,
        description="Boolean indicator: True if likely returned, False otherwise",
        examples=[True],
    )
    return_probability: float = Field(
        ...,
        description="Predicted probability score for return (0.0 to 1.0)",
        examples=[0.7245],
    )
    risk_level: str = Field(
        default="Low Risk",
        description="Categorical risk tier: High Risk (>=0.65), Moderate Risk (0.40-0.65), Low Risk (<0.40)",
        examples=["High Risk"],
    )
    model_name: str = Field(
        ...,
        description="Name of the model utilized for serving inference",
        examples=["LogisticRegression"],
    )
    engineered_features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Engineered features computed during preprocessing",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of the prediction",
    )


class BatchPredictionRequest(BaseModel):
    """Batch input payload."""

    orders: List[OrderReturnInput]


class BatchPredictionResponse(BaseModel):
    """Batch output response."""

    total_records: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """System health and status response."""

    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    model_name: str = Field(..., examples=["LogisticRegression"])
    app_version: str = Field(..., examples=["1.2.0"])
    ci_cd_pipeline: str = Field(
        default="Automated via GitHub Actions & GHCR",
        examples=["Automated via GitHub Actions & GHCR"],
    )
    environment: str = Field(default="production", examples=["production"])
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
