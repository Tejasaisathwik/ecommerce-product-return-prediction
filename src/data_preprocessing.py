"""Data preprocessing pipeline for the E-Commerce Product Return dataset."""

from __future__ import annotations
import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    return df


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Extract features and target, map target to binary (1=Returned, 0=Not Returned),

    and split into train and test sets.
    """
    raw_df = df.copy()

    # Target variable handling
    if "Return_Status" not in raw_df.columns:
        raise ValueError("Missing 'Return_Status' column in dataset.")

    y = (raw_df["Return_Status"].str.strip() == "Returned").astype(int)

    # Order_ID is an identifier, not a predictive feature
    drop_cols = ["Return_Status"]
    if "Order_ID" in raw_df.columns:
        drop_cols.append("Order_ID")

    X = raw_df.drop(columns=drop_cols)

    # Stratify only if each class has at least 2 samples
    class_counts = y.value_counts()
    can_stratify = (len(class_counts) > 1) and (class_counts.min() >= 2)
    strat = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    return X_train, X_test, y_train, y_test


def save_split_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = "data/processed"
) -> None:
    """Save processed train and test sets for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    train_df = X_train.copy()
    train_df["Return_Status"] = y_train.values

    test_df = X_test.copy()
    test_df["Return_Status"] = y_test.values

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
