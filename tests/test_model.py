"""Unit tests for feature engineering, preprocessing, and model pipelines."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import EcommerceFeatureEngineer
from src.data_preprocessing import prepare_data


@pytest.fixture
def sample_raw_dataframe():
    """Create sample dummy DataFrame representing e-commerce transactions."""
    return pd.DataFrame([
        {
            "Order_ID": "ORD001",
            "Product_Category": "Fashion",
            "Product_Price_INR": 15000.0,
            "Discount_Percent": 20.0,
            "Expected_Delivery_Days": 5,
            "Actual_Delivery_Days": 8,
            "Customer_Rating_PreOrder": 4.0,
            "Previous_Returns": 2,
            "Previous_Orders": 10,
            "Payment_Method": "Card",
            "Damaged_Package_Flag": "No",
            "Package_Size": "Medium",
            "Promotion_Active": "Yes",
            "Return_Status": "Returned",
        },
        {
            "Order_ID": "ORD002",
            "Product_Category": "Beauty",
            "Product_Price_INR": -100.0,  # Invalid price
            "Discount_Percent": 150.0,    # Invalid discount
            "Expected_Delivery_Days": 7,
            "Actual_Delivery_Days": 6,
            "Customer_Rating_PreOrder": 4.5,
            "Previous_Returns": 5,
            "Previous_Orders": 4,         # Previous returns > orders
            "Payment_Method": "COD",
            "Damaged_Package_Flag": "Yes",
            "Package_Size": "Small",
            "Promotion_Active": "No",
            "Return_Status": "Not Returned",
        },
    ])


def test_feature_engineer_calculations(sample_raw_dataframe):
    """Verify delivery delay, return frequency, discount calculations and invalid sanitization."""
    fe = EcommerceFeatureEngineer()
    transformed = fe.transform(sample_raw_dataframe)

    # First record checks:
    # delivery_delay = 8 - 5 = 3
    assert transformed.loc[0, "delivery_delay"] == 3.0
    assert transformed.loc[0, "is_delayed"] == 1.0
    # customer_return_frequency = 2 / 10 = 0.2
    assert transformed.loc[0, "customer_return_frequency"] == pytest.approx(0.2, abs=1e-4)
    # discount_amount = 15000 * 0.20 = 3000
    assert transformed.loc[0, "discount_amount"] == pytest.approx(3000.0, abs=1e-2)
    # effective_price = 15000 - 3000 = 12000
    assert transformed.loc[0, "effective_price"] == pytest.approx(12000.0, abs=1e-2)

    # Second record checks (invalid values handled gracefully):
    # Invalid negative price converted to NaN
    assert np.isnan(transformed.loc[1, "Product_Price_INR"])
    # Invalid discount > 100 converted to NaN
    assert np.isnan(transformed.loc[1, "Discount_Percent"])
    # delivery_delay = 6 - 7 = -1
    assert transformed.loc[1, "delivery_delay"] == -1.0
    assert transformed.loc[1, "is_delayed"] == 0.0


def test_prepare_data_split(sample_raw_dataframe):
    """Verify train/test splitting and target encoding."""
    X_train, X_test, y_train, y_test = prepare_data(sample_raw_dataframe, test_size=0.5, random_state=42)
    assert len(X_train) == 1
    assert len(X_test) == 1
    assert "Order_ID" not in X_train.columns
    assert "Return_Status" not in X_train.columns
    assert set(y_train.unique()).issubset({0, 1})
