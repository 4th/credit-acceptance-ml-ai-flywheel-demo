from textwrap import dedent


def generate_underwriter_summary(application: dict, model_outputs: dict) -> str:
    return dedent(f"""
    Application Summary (Demo)

    Dealer: {application.get('dealer_id')}
    Vehicle: {application.get('vehicle_type')} (age: {application.get('vehicle_age')})
    Income: ${application.get('applicant_income'):,.0f}
    Loan Amount: ${application.get('loan_amount'):,.0f}

    Model Output (Demo):
    - Estimated probability of default: {model_outputs.get('prob_bad', 0.5):.2f}
    - Suggested decision: {model_outputs.get('decision', 'REVIEW')}

    Rationale (Demo):
    - Higher loan amount relative to income increases risk.
    - Older vehicles tend to be associated with higher default probability.

    Next Suggested Actions (Demo):
    - Request additional income verification.
    - Consider alternative terms to reduce payment burden.
    """).strip()
