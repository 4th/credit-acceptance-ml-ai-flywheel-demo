import pandas as pd

CATEGORICAL_COLS = ["dealer_id", "vehicle_type", "region"]
NUMERIC_COLS = ["applicant_income", "loan_amount", "vehicle_age"]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["income_to_loan_ratio"] = df["applicant_income"] / (df["loan_amount"] + 1e-6)
    df["is_old_vehicle"] = (df["vehicle_age"] > 7).astype(int)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    return df
