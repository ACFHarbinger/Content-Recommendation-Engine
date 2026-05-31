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
    # Anthropic — required for query parsing (Phase 2) and explanations (Phase 5)
    anthropic_api_key: Optional[str] = Field(None, validation_alias="ANTHROPIC_API_KEY")

    # Qdrant — leave url blank to use local file-based mode
    qdrant_url: Optional[str] = Field(None, validation_alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(None, validation_alias="QDRANT_API_KEY")
    qdrant_local_path: str = Field(".qdrant_data", validation_alias="QDRANT_LOCAL_PATH")
    qdrant_collection: str = Field("listings", validation_alias="QDRANT_COLLECTION")

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

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

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
