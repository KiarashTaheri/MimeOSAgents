"""Runtime configuration, read from environment variables (prefix MIMOE_) or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MIMOE_", env_file=".env", extra="ignore")

    # mimOE exposes an OpenAI-compatible API; this is the only coupling point to the runtime.
    base_url: str = "http://localhost:8083/mimik-ai/openai/v1"
    # mimOE does not validate the key, but the OpenAI client requires one.
    api_key: str = "1234"
    # None -> use whichever model is currently loaded in mimOE Studio.
    model: str | None = None

    cgm_path: Path = PROJECT_ROOT / "data" / "cgm_mock.csv"

    max_steps: int = Field(default=6, ge=1, description="Max LLM calls per user turn.")
    max_retries: int = Field(default=3, ge=0, description="HTTP-level retries (SDK backoff).")
    timeout_s: float = Field(default=120.0, gt=0)
    temperature: float = Field(default=0.2, ge=0, le=2)
    # Qwen3-style reasoning models think by default; off keeps latency low on-device.
    thinking: bool = False

    # Logfire write token (no MIMOE_ prefix). Unset -> traces stay local, nothing is exported.
    logfire_token: SecretStr | None = Field(default=None, validation_alias="LOGFIRE_TOKEN")


@lru_cache
def get_settings() -> Settings:
    return Settings()
