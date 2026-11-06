"""Feature engineering and data transformation components for E-Commerce Product Return Prediction."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_COLUMNS = [
    "Product_Price_INR",
    "Discount_Percent",
    "Expected_Delivery_Days",
    "Actual_Delivery_Days",
    "Customer_Rating_PreOrder",
    "Previous_Returns",
    "Previous_Orders",
    "delivery_delay",
    "customer_return_frequency",
    "discount_amount",
    "effective_price",
    "is_delayed",
]

CATEGORICAL_COLUMNS = [
    "Product_Category",
    "Payment_Method",
    "Damaged_Package_Flag",
    "Package_Size",
    "Promotion_Active",
]

TARGET_COLUMN = "Return_Status"


class EcommerceFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to sanitize invalid values and compute domain features."""

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Sanitize invalid product prices (e.g. <= 0)
        if "Product_Price_INR" in df.columns:
            df["Product_Price_INR"] = pd.to_numeric(df["Product_Price_INR"], errors="coerce")
            df.loc[df["Product_Price_INR"] <= 0, "Product_Price_INR"] = np.nan

        # Sanitize invalid discount percentages (< 0 or > 100)
        if "Discount_Percent" in df.columns:
            df["Discount_Percent"] = pd.to_numeric(df["Discount_Percent"], errors="coerce")
            df.loc[(df["Discount_Percent"] < 0) | (df["Discount_Percent"] > 100), "Discount_Percent"] = np.nan

        # Sanitize delivery days
        if "Expected_Delivery_Days" in df.columns:
            df["Expected_Delivery_Days"] = pd.to_numeric(df["Expected_Delivery_Days"], errors="coerce")
            df.loc[df["Expected_Delivery_Days"] < 0, "Expected_Delivery_Days"] = np.nan

        if "Actual_Delivery_Days" in df.columns:
            df["Actual_Delivery_Days"] = pd.to_numeric(df["Actual_Delivery_Days"], errors="coerce")
            df.loc[df["Actual_Delivery_Days"] < 0, "Actual_Delivery_Days"] = np.nan

        if "Customer_Rating_PreOrder" in df.columns:
            df["Customer_Rating_PreOrder"] = pd.to_numeric(df["Customer_Rating_PreOrder"], errors="coerce")
            invalid_rating = (df["Customer_Rating_PreOrder"] < 1.0) | (df["Customer_Rating_PreOrder"] > 5.0)
            df.loc[invalid_rating, "Customer_Rating_PreOrder"] = np.nan

        if "Previous_Returns" in df.columns:
            df["Previous_Returns"] = pd.to_numeric(df["Previous_Returns"], errors="coerce")
            df.loc[df["Previous_Returns"] < 0, "Previous_Returns"] = np.nan

        if "Previous_Orders" in df.columns:
            df["Previous_Orders"] = pd.to_numeric(df["Previous_Orders"], errors="coerce")
            df.loc[df["Previous_Orders"] < 0, "Previous_Orders"] = np.nan

        # Cap previous returns to previous orders if illogical
        if "Previous_Returns" in df.columns and "Previous_Orders" in df.columns:
            invalid_ret = df["Previous_Returns"] > df["Previous_Orders"]
            df.loc[invalid_ret, "Previous_Returns"] = df.loc[invalid_ret, "Previous_Orders"]

        # Feature Engineering:
        # 1. Delivery delay (days difference between actual and expected delivery)
        if "Actual_Delivery_Days" in df.columns and "Expected_Delivery_Days" in df.columns:
            df["delivery_delay"] = df["Actual_Delivery_Days"] - df["Expected_Delivery_Days"]
            df["is_delayed"] = (df["delivery_delay"] > 0).astype(float)
        else:
            df["delivery_delay"] = np.nan
            df["is_delayed"] = np.nan

        # 2. Customer return frequency (ratio of past returns to past orders)
        if "Previous_Returns" in df.columns and "Previous_Orders" in df.columns:
            orders_denom = df["Previous_Orders"].replace(0, 1)
            df["customer_return_frequency"] = (df["Previous_Returns"] / orders_denom).clip(0.0, 1.0)
        else:
            df["customer_return_frequency"] = np.nan

        # 3. Discount amount and effective paid price
        if "Product_Price_INR" in df.columns and "Discount_Percent" in df.columns:
            df["discount_amount"] = df["Product_Price_INR"] * (df["Discount_Percent"] / 100.0)
            df["effective_price"] = df["Product_Price_INR"] - df["discount_amount"]
        else:
            df["discount_amount"] = np.nan
            df["effective_price"] = np.nan

        # Filter only defined features
        available_features = [col for col in (NUMERIC_COLUMNS + CATEGORICAL_COLUMNS) if col in df.columns]
        return df[available_features]
