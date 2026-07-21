"""
train.py
--------
Train baseline models for churn prediction, with MLflow experiment tracking.

Usage (from project root, with venv activated):
    python src/train.py
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

PROCESSED_DATA_PATH = Path("data/processed/churn_clean.csv")
MLFLOW_EXPERIMENT_NAME = "churn-prediction-baseline"


def load_processed_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def split_data(X, y):
    # stratify=y keeps the same churn ratio in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Train churn ratio: {y_train.mean():.3f} | Test churn ratio: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def balance_classes(X_train, y_train):
    """
    Apply SMOTE to the TRAINING set only.
    Never apply SMOTE to the test set - that would leak synthetic
    data into your evaluation and give you falsely good metrics.
    """
    print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE:  {y_train_bal.value_counts().to_dict()}")
    return X_train_bal, y_train_bal


def evaluate_model(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n--- {model_name} ---")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


def train_and_log(model, model_name: str, X_train, y_train, X_test, y_test, params: dict):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, model_name)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    return model, metrics


def run_pipeline():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)

    results = {}

    # --- Model 1: Logistic Regression ---
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    _, metrics_lr = train_and_log(
        log_reg, "logistic_regression",
        X_train_bal, y_train_bal, X_test, y_test,
        params={"model_type": "LogisticRegression", "balancing": "SMOTE", "max_iter": 1000},
    )
    results["logistic_regression"] = metrics_lr

    # --- Model 2: Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    _, metrics_rf = train_and_log(
        rf, "random_forest",
        X_train_bal, y_train_bal, X_test, y_test,
        params={"model_type": "RandomForestClassifier", "balancing": "SMOTE",
                "n_estimators": 200, "max_depth": 10},
    )
    results["random_forest"] = metrics_rf

    # --- Summary comparison ---
    print("\n" + "=" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 50)
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(4))

    print("\nTo view detailed experiment comparisons, run: mlflow ui")
    print("Then open http://localhost:5000 in your browser")

    return results


if __name__ == "__main__":
    run_pipeline()