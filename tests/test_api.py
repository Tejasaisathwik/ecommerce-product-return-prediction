"""Integration tests for FastAPI endpoints."""

from __future__ import annotations
from fastapi.testclient import TestClient

from app.main import app, load_artifacts

# Ensure artifacts are loaded for tests
load_artifacts()
client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint provides API details."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "E-Commerce Product Return Prediction API"
    assert data["status"] == "online"


def test_health_endpoint():
    """Verify /health endpoint returns healthy status and model information."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_name" in data


def test_predict_endpoint_valid():
    """Verify /predict endpoint returns proper prediction structure."""
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Returned", "Not Returned"]
    assert isinstance(data["is_returned"], bool)
    assert 0.0 <= data["return_probability"] <= 1.0
    assert "engineered_features" in data
    assert data["engineered_features"]["delivery_delay_days"] == 3.0
    assert data["engineered_features"]["is_delayed"] is True


def test_predict_endpoint_invalid_input():
    """Verify /predict rejects invalid payload with HTTP 422."""
    bad_payload = {
        "Product_Category": "Fashion",
        "Product_Price_INR": -50.0,  # Invalid: gt 0 constraint
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_batch_predict_endpoint():
    """Verify /batch-predict processes multiple records."""
    batch_payload = {
        "orders": [
            {
                "Product_Category": "Electronics",
                "Product_Price_INR": 24999.0,
                "Discount_Percent": 10.0,
                "Expected_Delivery_Days": 4,
                "Actual_Delivery_Days": 4,
                "Customer_Rating_PreOrder": 4.5,
                "Previous_Returns": 0,
                "Previous_Orders": 15,
                "Payment_Method": "Card",
                "Damaged_Package_Flag": "No",
                "Package_Size": "Small",
                "Promotion_Active": "No",
            },
            {
                "Product_Category": "Beauty",
                "Product_Price_INR": 2500.0,
                "Discount_Percent": 50.0,
                "Expected_Delivery_Days": 3,
                "Actual_Delivery_Days": 7,
                "Customer_Rating_PreOrder": 2.5,
                "Previous_Returns": 5,
                "Previous_Orders": 6,
                "Payment_Method": "UPI",
                "Damaged_Package_Flag": "Yes",
                "Package_Size": "Large",
                "Promotion_Active": "Yes",
            },
        ]
    }
    response = client.post("/batch-predict", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2


def test_metrics_endpoint():
    """Verify /metrics returns evaluation comparison."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "best_model" in data
    assert "models" in data
