# Customer Churn Prediction

End-to-end ML project: predicts whether a telecom customer will churn, explains why (SHAP),
and serves predictions through an API + Streamlit app.

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install libomp (required for lightgbm on Mac)
brew install libomp

# 3. Install dependencies
pip install -r requirements.txt
```

## Get the dataset

1. Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Unzip and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` inside `data/raw/`

## Step 1 — EDA

Open `notebooks/01_eda.py` in VS Code (select your venv as the interpreter first).
Run each `# %%` cell to explore the data.

## Step 2 — Preprocessing

```bash
python src/preprocessing.py
```

This cleans the raw data (fixes `TotalCharges`, encodes categoricals) and saves
`data/processed/churn_clean.csv`.

## Next steps (coming up in the roadmap)

- `src/train.py` — train and compare baseline models (Logistic Regression, Random Forest)
- MLflow experiment tracking
- XGBoost/LightGBM tuning
- SHAP explainability
- FastAPI + Streamlit app
- Docker + deployment

See the full week-by-week plan in `Customer_Churn_Project_Workflow.pdf`.
