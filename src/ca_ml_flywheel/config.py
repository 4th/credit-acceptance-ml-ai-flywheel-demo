from pydantic import BaseSettings


class Settings(BaseSettings):
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name_approval: str = "credit_approval_xgb"
    model_stage: str = "Production"

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"


settings = Settings()
