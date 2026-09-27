from functools import lru_cache
from pathlib import Path


from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    milvus_uri: str = Field(min_length=1)

    mongo_host: str = Field(min_length=1)
    mongo_port: int = Field(ge=1, le=65535)
    mongo_username: str = Field(min_length=1)
    mongo_password: SecretStr = Field(min_length=1)
    mongo_auth_source: str = Field(min_length=1)

    minio_endpoint: str = Field(min_length=1)
    minio_access_key: str = Field(min_length=1)
    minio_secret_key: SecretStr = Field(min_length=1)
    minio_secure: bool
    minio_bucket_name: str = Field(
        min_length=3,
        max_length=63,
        pattern=r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

