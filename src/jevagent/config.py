"""Settings loaded from environment / .env."""

from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["dev", "staging", "prod"] = "dev"
    jev_model: str = "jev-1.14.0"
    jev_api_key: SecretStr = SecretStr("")
    jev_base_url: str = "https://api.typesafe.ai"
    jev_timeout_s: float = 2.0
    jev_max_retries: int = 1
    # Early-handoff bias (H-2): confidence below this after one re-ask hands off.
    default_confidence_threshold: float = 0.7
    max_retries_per_step: int = 1

    @model_validator(mode="after")
    def _pin_model(self) -> "Settings":
        if self.app_env != "dev" and self.jev_model.endswith("-latest"):
            raise ValueError(
                f"jev_model={self.jev_model!r} is not allowed when app_env={self.app_env!r}; "
                "pin an exact version such as jev-1.14.0 (A-11)"
            )
        return self
