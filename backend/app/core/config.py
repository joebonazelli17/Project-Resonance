from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Resonance"
    DEBUG: bool = False

    # Postgres
    DATABASE_URL: str = "postgresql+asyncpg://resonance:resonance@db:5432/resonance"

    # S3 / MinIO
    S3_ENDPOINT: str = "http://minio:9000"
    S3_PUBLIC_ENDPOINT: str = "http://127.0.0.1:9002"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_TRACKS: str = "resonance-tracks"
    S3_REGION: str = "us-east-1"

    # CLAP model
    CLAP_CKPT: str = "/models/clap/music_audioset_epoch_15_esc_90.14.pt"
    CLAP_DEVICE: str = "cpu"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Analysis defaults
    DEFAULT_BARS_LIST: list[int] = [2, 4, 8, 16]
    DEFAULT_HOP_BARS: int = 2
    DEFAULT_BEATS_PER_BAR: int = 4

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
