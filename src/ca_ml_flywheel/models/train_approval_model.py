import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

from ca_ml_flywheel.config import settings
from ca_ml_flywheel.features.credit_features import build_feature_matrix

# Project root assumed to be repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """Fallback: generate synthetic used-car finance applications (demo only)."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "dealer_id": rng.integers(1, 50, size=n_samples).astype(str),
        "vehicle_type": rng.choice(["car", "truck", "suv"], size=n_samples),
        "region": rng.choice(["midwest", "south", "west"], size=n_samples),
        "applicant_income": rng.normal(45000, 15000, size=n_samples),
        "loan_amount": rng.normal(16000, 5000, size=n_samples).clip(5000, 40000),
        "vehicle_age": rng.integers(0, 15, size=n_samples),
    })
    risk_score = (
        0.00005 * df["loan_amount"]
        - 0.00003 * df["applicant_income"]
        + 0.05 * df["vehicle_age"]
    )
    prob_bad = 1 / (1 + np.exp(-risk_score))
    df["bad"] = (rng.random(n_samples) < prob_bad).astype(int)
    return df


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Try to load processed train/test CSVs from data/processed.
    If they don't exist, raise FileNotFoundError.
    """
    train_path = DATA_PROCESSED_DIR / "credit_approval_train.csv"
    test_path = DATA_PROCESSED_DIR / "credit_approval_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected processed files not found in {DATA_PROCESSED_DIR}. "
            f"Missing one of: {train_path.name}, {test_path.name}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def main():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("credit_approval_xgb_demo")

    use_processed = True
    try:
        train_df, test_df = load_processed_data()
        print(f"Loaded processed datasets from {DATA_PROCESSED_DIR}")
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        print("[INFO] Falling back to synthetic data generation.")
        df = load_synthetic_data()
        train_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df["bad"], random_state=42
        )
        use_processed = False

    # Build features
    X_train = build_feature_matrix(train_df)
    y_train = train_df["bad"]
    X_test = build_feature_matrix(test_df)
    y_test = test_df["bad"]

    with mlflow.start_run():
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("data_source", "processed_csv" if use_processed else "synthetic")

        # Log model to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Also save a local copy for the FastAPI service
        local_model_path = MODELS_DIR / "credit_approval_xgb.joblib"
        joblib.dump(model, local_model_path)
        print(f"Saved trained model to {local_model_path}")

        print(f"Trained credit approval model, AUC = {auc:.3f}")


if __name__ == "__main__":
    main()
