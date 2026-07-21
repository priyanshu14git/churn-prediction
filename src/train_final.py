"""
train_final.py
---------------
Week 4: Train the final chosen model (LightGBM, best on recall),
save it to disk, and add SHAP explainability.

Usage (from project root, with venv activated):
    python src/train_final.py
"""

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

PROCESSED_DATA_PATH = Path("data/processed/churn_clean.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "final_model.pkl"
SHAP_SUMMARY_PATH = MODEL_DIR / "shap_summary_plot.png"

# Best params found from your GridSearchCV run - update these if yours differ
BEST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "random_state": 42,
}


def load_processed_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def train_final_model():
    X, y = load_processed_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print("Training final LightGBM model with best params:", BEST_PARAMS)
    model = LGBMClassifier(**BEST_PARAMS, verbose=-1)
    model.fit(X_train_bal, y_train_bal)

    # Final sanity-check metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\nFinal model performance on test set:")
    print(f"accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"precision: {precision_score(y_test, y_pred):.4f}")
    print(f"recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"f1       : {f1_score(y_test, y_pred):.4f}")
    print(f"roc_auc  : {roc_auc_score(y_test, y_proba):.4f}")

    # Save model + feature column order (need this later for the API)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, MODEL_PATH)
    print(f"\nSaved final model to {MODEL_PATH}")

    return model, X_train_bal, X_test, y_test


def explain_with_shap(model, X_train_bal, X_test):
    """
    Generate SHAP explanations:
    1. A global summary plot - which features matter most overall
    2. A single-prediction explanation - why one specific customer
       was predicted to churn or not
    """
    print("\nComputing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # --- Global summary plot ---
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {SHAP_SUMMARY_PATH}")

    # --- Single customer explanation (first row of test set, as an example) ---
    sample_idx = 0
    sample = X_test.iloc[[sample_idx]]
    sample_shap = explainer.shap_values(sample)

    print(f"\nExample: explanation for customer #{sample_idx} in the test set")
    print(f"Predicted churn probability: {model.predict_proba(sample)[0][1]:.3f}")

    # Print top 5 features driving this specific prediction
    feature_impact = pd.Series(sample_shap[0], index=sample.columns).sort_values(
        key=abs, ascending=False
    )
    print("\nTop 5 features driving this prediction:")
    print(feature_impact.head(5))

    return explainer, shap_values


if __name__ == "__main__":
    model, X_train_bal, X_test, y_test = train_final_model()
    explain_with_shap(model, X_train_bal, X_test)