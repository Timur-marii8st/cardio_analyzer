from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
import secrets

class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # API
    artifacts_path: str = Field(
        default="packages/ctg_ml/patient_risk_pipeline.pkl",
        description="Path to ML model artifact (pkl or joblib)"
    )
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    
    # Database
    database_url: str = Field(
        ...,  # Required!
        description="PostgreSQL connection string"
    )
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string"
    )
    
    # Security - MUST be set in production
    secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for JWT tokens"
    )
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=60)
    refresh_token_expire_days: int = Field(default=7)
    
    # CORS - comma-separated origins
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        description="Allowed CORS origins"
    )
    
    @validator("secret_key")
    def validate_secret_key(cls, v):
        if v == "change-this-secret-key-in-production":
            raise ValueError(
                "SECRET_KEY must be changed from default value in production!"
            )
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v
    
    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

api_settings = ApiSettings()