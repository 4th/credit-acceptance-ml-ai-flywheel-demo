from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from ca_ml_flywheel.models.train_approval_model import main as train_approval_model


default_args = {
    "owner": "ml_team",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


with DAG(
    dag_id="credit_approval_training_daily",
    default_args=default_args,
    schedule_interval="0 3 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "credit_approval"],
) as dag:

    train_credit_approval = PythonOperator(
        task_id="train_credit_approval_model",
        python_callable=train_approval_model,
    )
