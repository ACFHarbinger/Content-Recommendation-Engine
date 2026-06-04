"""
Settings loaded from environment variables / .env file.
All fields have sensible defaults so the engine works locally without any config.
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Anthropic — required for query parsing (Phase 2) and explanations (Phase 5)
    anthropic_api_key: Optional[str] = Field(None, validation_alias="ANTHROPIC_API_KEY")

    # SQLite storage path (use ":memory:" for tests)
    sqlite_path: str = Field("data/rec_engine.db", validation_alias="SQLITE_PATH")

    # Embedding (Phase 1)
    embed_model: str = Field("BAAI/bge-m3", validation_alias="EMBED_MODEL")
    default_top_k: int = Field(10, validation_alias="DEFAULT_TOP_K")
    batch_size: int = Field(32, validation_alias="BATCH_SIZE")

    # Scoring decay (Phase 4)
    lambda_recency: float = Field(0.05, validation_alias="LAMBDA_RECENCY", ge=0.0, le=1.0)
    length_origin: int = Field(12, validation_alias="LENGTH_ORIGIN", ge=1)
    length_scale: int = Field(24, validation_alias="LENGTH_SCALE", ge=1)

    # Fusion strategy: "rrf" or "dbsf" (Phase 3)
    fusion_method: str = Field("rrf", validation_alias="FUSION_METHOD")

    # Reranker (Phase 8)
    rerank_model: str = Field("BAAI/bge-reranker-v2-m3", validation_alias="RERANK_MODEL")
    rerank_top_n: int = Field(20, validation_alias="RERANK_TOP_N")

    # Claude model for query parser & explainer (Phases 2, 5)
    claude_model: str = Field("claude-sonnet-4-6", validation_alias="CLAUDE_MODEL")

    # Watch-history implicit feedback (Phase 9)
    history_min_rating: float = Field(7.0, validation_alias="HISTORY_MIN_RATING", ge=0.0, le=10.0)
    history_boost_weight: float = Field(0.15, validation_alias="HISTORY_BOOST_WEIGHT", ge=0.0, le=1.0)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
