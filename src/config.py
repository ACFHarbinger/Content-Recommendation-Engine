"""
Settings loaded from environment variables / .env file.
All fields have sensible defaults so the engine works locally without any config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")

    # Qdrant — leave url blank to use local file-based mode
    qdrant_url: Optional[str] = Field(None, env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    qdrant_local_path: str = Field(".qdrant_data", env="QDRANT_LOCAL_PATH")
    qdrant_collection: str = Field("listings", env="QDRANT_COLLECTION")

    # Embedding
    embed_model: str = Field("BAAI/bge-m3", env="EMBED_MODEL")

    # Retrieval
    default_top_k: int = Field(10, env="DEFAULT_TOP_K")
    batch_size: int = Field(32, env="BATCH_SIZE")

    # Scoring decay
    lambda_recency: float = Field(0.05, env="LAMBDA_RECENCY", ge=0.0, le=1.0)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def use_local_qdrant(self) -> bool:
        return not bool(self.qdrant_url)

    @property
    def qdrant_storage_path(self) -> Optional[str]:
        return str(Path(self.qdrant_local_path).resolve()) if self.use_local_qdrant else None


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
