"""
Face Recognition Service — Application Configuration

All settings are loaded from environment variables via pydantic-settings.
See .env.example for the full list of configurable values.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "face-service"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "info"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Database ---
    database_url: str = "postgresql+psycopg://faceuser:facepass@db:5432/facedb"

    # --- Internal API Security ---
    api_secret_key: str = "change-me-to-a-long-random-string"

    # --- Face Pipeline ---
    face_model_pretrained: str = "vggface2"
    face_detect_threshold: float = 0.95
    face_verify_threshold: float = 0.75
    face_max_faces_enroll: int = 1
    face_min_face_size: int = 80
    max_image_size: int = 5_242_880  # 5 MB

    # --- Image Storage ---
    image_storage_path: str = "/app/storage/faces"

    # --- Recognition ---
    recognize_top_k: int = 5

    # --- OpenAPI ---
    openapi_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — parsed once on first access."""
    return Settings()
