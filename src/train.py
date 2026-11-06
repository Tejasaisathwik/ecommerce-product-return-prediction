"""Model training, evaluation, comparison, and serialization script."""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / CI environments
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_preprocessing import load_raw_data, prepare_data, save_split_data
from src.feature_engineering import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    EcommerceFeatureEngineer,
)


def build_preprocessor() -> ColumnTransformer:
    """Construct column transformer with imputers, scalers, and encoders."""
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERIC_COLUMNS),
            ("cat", cat_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
    return preprocessor


def build_full_pipeline(classifier) -> Pipeline:
    """Build complete end-to-end pipeline from raw DataFrame to prediction."""
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("feature_engineer", EcommerceFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def train_and_evaluate(data_path: str = "data/raw/ecommerce_returns.csv", output_dir: str = "models"):
    """Main training routine."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Loading raw dataset from: {data_path}")
    raw_df = load_raw_data(data_path)

    print("[*] Preparing train/test split...")
    X_train, X_test, y_train, y_test = prepare_data(raw_df, test_size=0.2, random_state=42)
    save_split_data(X_train, X_test, y_train, y_test)
    print(f"    Train size: {len(X_train)} | Test size: {len(X_test)}")

    candidate_models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
    }

    results = {}
    fitted_pipelines = {}
    roc_curves = {}

    for name, clf in candidate_models.items():
        print(f"\n[*] Training candidate model: {name}...")
        pipeline = build_full_pipeline(clf)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr)

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc, 4),
            "confusion_matrix": cm,
        }
        fitted_pipelines[name] = pipeline

        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        print(f"    ROC-AUC:   {roc:.4f}")

    # Determine best model based on composite score (F1 + ROC-AUC)
    # Balanced LogisticRegression / RandomForest handle imbalanced returns well
    best_model_name = max(
        results.keys(),
        key=lambda k: results[k]["roc_auc"] + results[k]["f1_score"]
    )
    print(f"\n[+] Selected Best Model: {best_model_name}")

    best_pipeline = fitted_pipelines[best_model_name]
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"[+] Saved serialized pipeline to: {best_model_path}")

    # Save metrics JSON
    metrics_summary = {
        "best_model": best_model_name,
        "models": results,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[+] Saved metrics report to: {metrics_path}")

    # Generate Model Comparison Chart
    _generate_comparison_chart(results, os.path.join(output_dir, "model_comparison.png"))

    # Generate Confusion Matrix Chart
    _generate_confusion_matrix_chart(
        np.array(results[best_model_name]["confusion_matrix"]),
        best_model_name,
        os.path.join(output_dir, "confusion_matrix.png"),
    )

    # Generate ROC Curves
    _generate_roc_chart(roc_curves, results, os.path.join(output_dir, "roc_curve.png"))

    return metrics_summary


def _generate_comparison_chart(results: dict, output_file: str):
    """Plot bar chart comparing model metrics."""
    models = list(results.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    x = np.arange(len(models))
    width = 0.15

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics_names):
        values = [results[m][metric] for m in models]
        plt.bar(x + i * width, values, width, label=metric.replace("_", " ").title())

    plt.xlabel("Candidate Models", fontweight="bold", fontsize=11)
    plt.ylabel("Score", fontweight="bold", fontsize=11)
    plt.title("E-Commerce Return Prediction - Model Performance Comparison", fontweight="bold", fontsize=13)
    plt.xticks(x + width * 2, models, fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"[+] Comparison chart saved to {output_file}")


def _generate_confusion_matrix_chart(cm: np.ndarray, model_name: str, output_file: str):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}", fontweight="bold", fontsize=12)
    plt.colorbar()
    classes = ["Not Returned (0)", "Returned (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=9)
    plt.yticks(tick_marks, classes, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold", fontsize=12
            )

    plt.ylabel("True Class", fontweight="bold")
    plt.xlabel("Predicted Class", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"[+] Confusion matrix chart saved to {output_file}")


def _generate_roc_chart(roc_curves: dict, results: dict, output_file: str):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr) in roc_curves.items():
        auc_val = results[name]["roc_auc"]
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"[+] ROC curve chart saved to {output_file}")


if __name__ == "__main__":
    train_and_evaluate()
