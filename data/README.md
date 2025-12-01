# Data Directory

## raw/

Source tables for the demo credit / auto finance project. Example files:

- `applications.csv` — application-level data  
  - Columns: `application_id, dealer_id, vehicle_type, region, applicant_income, loan_amount, vehicle_age, application_date`
- `payments.csv` — payment history / performance data  
  - Columns: `application_id, payment_date, amount, status`
- `dealers.csv` — dealer metadata  
  - Columns: `dealer_id, name, region, volume_tier, risk_flag`
- `vehicles.csv` — vehicle attributes  
  - Columns: `vin, vehicle_type, make, model, year, mileage`

In a real system these would come from a warehouse or lake (e.g., Databricks, Snowflake)
and may be larger and partitioned.

## processed/

Cleaned, transformed, model-ready data. Example placeholders:

- `credit_approval_train.parquet` — feature matrix + labels for training
- `credit_approval_test.parquet` — held-out evaluation data
- `lifecycle_uplift_dataset.parquet` — dataset prepared for causal uplift modeling
- `dealer_bandit_context.parquet` — contextual features for dealer bandit policy

These are typically derived from `raw/` by your feature engineering pipelines.

## models/ (optional)

In your main project, you might also keep serialized models here, such as:

- `credit_approval_xgb.joblib`

For this demo, models are created by the training scripts and can be stored
wherever your configuration points (e.g., `data/models/` or MLflow artifact store).
