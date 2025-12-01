import pandas as pd
from ca_ml_flywheel.features.credit_features import build_feature_matrix


def test_build_feature_matrix_basic():
    df = pd.DataFrame(
        [{
            "dealer_id": "1",
            "vehicle_type": "car",
            "region": "south",
            "applicant_income": 50000,
            "loan_amount": 15000,
            "vehicle_age": 3,
        }]
    )
    X = build_feature_matrix(df)
    assert "income_to_loan_ratio" in X.columns
    assert "is_old_vehicle" in X.columns
    assert len(X) == 1
