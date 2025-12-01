from pathlib import Path
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ca_ml_flywheel.features.credit_features import build_feature_matrix
from ca_ml_flywheel.models.bandit_recommender import DealerBandit

app = FastAPI(title="Credit Acceptance ML & Gen-AI Demo API")

# -----------------------------
# Paths / Model Loading
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Credit approval model
DEFAULT_MODEL_PATH = DATA_DIR / "models" / "credit_approval_xgb.joblib"
MODEL_PATH = Path(os.getenv("APPROVAL_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

# Dealer bandit
BANDIT_CONTEXT_PATH = PROCESSED_DIR / "dealer_bandit_context.csv"

if BANDIT_CONTEXT_PATH.exists():
    bandit = DealerBandit.from_historical_data(path=BANDIT_CONTEXT_PATH)
else:
    # Fallback: start with empty bandit and default arms
    bandit = DealerBandit()


# -----------------------------
# Schemas
# -----------------------------

class ApplicationRequest(BaseModel):
    dealer_id: str
    vehicle_type: str
    region: str
    applicant_income: float
    loan_amount: float
    vehicle_age: int


class ApprovalPrediction(BaseModel):
    approval_risk: float
    decision: str


class DealerContextRequest(BaseModel):
    dealer_id: str


class DealerRecommendationResponse(BaseModel):
    dealer_id: str
    recommended_layout: str
    # optional: simple debug info for demo purposes
    note: str | None = None


class BanditFeedbackRequest(BaseModel):
    dealer_id: str
    arm_id: str
    reward: int  # 1 for success (e.g., click), 0 for no click


# -----------------------------
# Endpoints
# -----------------------------

@app.post("/predict/approval", response_model=ApprovalPrediction)
def predict_approval(req: ApplicationRequest):
    """
    Predict credit approval risk for a given application.
    """
    if model is None:
        return ApprovalPrediction(
            approval_risk=0.5,
            decision="MODEL_UNAVAILABLE_DEMO_ONLY",
        )

    df = pd.DataFrame([req.dict()])
    X = build_feature_matrix(df)
    prob_bad = float(model.predict_proba(X)[:, 1])

    decision = "APPROVE" if prob_bad < 0.4 else "REVIEW"

    return ApprovalPrediction(approval_risk=prob_bad, decision=decision)


@app.post("/recommend/dealer-next-action", response_model=DealerRecommendationResponse)
def recommend_dealer_next_action(req: DealerContextRequest):
    """
    Use a contextual bandit to select a layout / next-best-action arm for a dealer.

    For demo purposes, the "arm" is just a layout string like 'layout_a', 'layout_b', etc.
    """
    recommended_arm = bandit.select_arm(req.dealer_id)

    return DealerRecommendationResponse(
        dealer_id=req.dealer_id,
        recommended_layout=recommended_arm,
        note="Thompson Sampling-based recommendation (demo).",
    )


@app.post("/bandit/update")
def update_bandit_feedback(req: BanditFeedbackRequest):
    """
    Update bandit statistics based on observed reward.

    Example: after showing a layout to a dealer, if they clicked:
    - reward = 1
    otherwise:
    - reward = 0
    """
    bandit.update(dealer_id=req.dealer_id, arm_id=req.arm_id, reward=req.reward)
    return {"status": "ok", "message": "Bandit updated."}
