"""
train_advanced.py
------------------
Week 3: Train XGBoost and LightGBM with hyperparameter tuning,
logged to the same MLflow experiment as the baseline models.

Usage (from project root, with venv activated):
    python src/train_advanced.py
"""

import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROCESSED_DATA_PATH = Path("data/processed/churn_clean.csv")
MLFLOW_EXPERIMENT_NAME = "churn-prediction-baseline"  # same experiment as train.py


def load_processed_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def balance_classes(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)


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


def tune_and_log(base_model, param_grid, model_name, X_train, y_train, X_test, y_test, log_model_fn):
    """
    Runs GridSearchCV to find the best hyperparameters, then logs the
    best model + its params + metrics to MLflow.

    log_model_fn: the correct MLflow logging function for this model type
                  (e.g. mlflow.xgboost.log_model or mlflow.lightgbm.log_model).
                  Using the generic mlflow.sklearn.log_model on XGBoost/LightGBM
                  models triggers a skops "untrusted type" error - these libraries
                  have their own dedicated MLflow flavors instead.
    """
    print(f"\nRunning GridSearchCV for {model_name}... (this may take a few minutes)")

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="recall",  # optimize for recall - our key business metric
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\nBest params for {model_name}: {grid_search.best_params_}")

    with mlflow.start_run(run_name=f"{model_name}_tuned"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("balancing", "SMOTE")
        mlflow.log_param("tuning", "GridSearchCV (optimized for recall)")

        metrics = evaluate_model(best_model, X_test, y_test, f"{model_name} (tuned)")
        mlflow.log_metrics(metrics)
        log_model_fn(best_model, "model")

    return best_model, metrics


def run_pipeline():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)

    results = {}

    # --- XGBoost ---
    xgb_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
    }
    xgb_base = XGBClassifier(
        random_state=42, eval_metric="logloss", use_label_encoder=False
    )
    _, xgb_metrics = tune_and_log(
        xgb_base, xgb_param_grid, "xgboost",
        X_train_bal, y_train_bal, X_test, y_test,
        log_model_fn=mlflow.xgboost.log_model,
    )
    results["xgboost_tuned"] = xgb_metrics

    # --- LightGBM ---
    lgbm_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
    }
    lgbm_base = LGBMClassifier(random_state=42, verbose=-1)
    _, lgbm_metrics = tune_and_log(
        lgbm_base, lgbm_param_grid, "lightgbm",
        X_train_bal, y_train_bal, X_test, y_test,
        log_model_fn=mlflow.lightgbm.log_model,
    )
    results["lightgbm_tuned"] = lgbm_metrics

    # --- Summary ---
    print("\n" + "=" * 60)
    print("ADVANCED MODEL COMPARISON SUMMARY")
    print("=" * 60)
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(4))

    best_model_name = comparison_df["recall"].idxmax()
    print(f"\nBest model by recall: {best_model_name}")
    print("Run 'mlflow ui --port 5001' and open the experiment to compare ALL runs "
          "(baseline + tuned) side by side.")

    return results


if __name__ == "__main__":
    run_pipeline()