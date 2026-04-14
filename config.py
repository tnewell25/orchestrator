from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    agent_name: str = "Orchestrator"
    environment: str = "production"
    debug: bool = False
    port: int = 8000

    # LLM — OAuth uses your Claude Max subscription (no API costs)
    anthropic_api_key: str = ""
    anthropic_auth_token: str = ""
    anthropic_refresh_token: str = ""
    default_model: str = "claude-sonnet-4-5-20250929"
    fast_model: str = "claude-haiku-4-5-20251001"

    # Embeddings (local via fastembed)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Database
    database_url: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_allowed_users: str = ""  # comma-separated user IDs

    # Voice transcription
    openai_api_key: str = ""

    # Google (Gmail + Calendar)
    google_credentials_path: str = ""
    google_user_email: str = ""

    # Research (optional — Serper for cleaner google results; DDG scraping fallback)
    serper_api_key: str = ""

    # Proactive monitor tuning
    stalled_deal_days: int = 14
    unanswered_email_days: int = 5

    @property
    def allowed_user_ids(self) -> list[int]:
        if not self.telegram_allowed_users:
            return []
        return [int(uid.strip()) for uid in self.telegram_allowed_users.split(",") if uid.strip()]

    def validate_critical(self) -> list[str]:
        warnings = []
        if not self.database_url:
            warnings.append("DATABASE_URL not set — memory will not persist")
        if not self.anthropic_api_key and not self.anthropic_auth_token:
            warnings.append("No LLM credentials — set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN")
        if not self.telegram_bot_token:
            warnings.append("TELEGRAM_BOT_TOKEN not set — agent has no interface")
        return warnings

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
