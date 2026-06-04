"""
Shared test fixtures for the Recommendation Engine test suite.

Provides:
  cfg               — Settings with a temp-file SQLite path (isolated per test)
  sqlite_store      — A ready-to-use SQLiteStore backed by the cfg db
  anthropic_mock    — Injects a minimal async-capable fake anthropic module
  mock_embedder     — MockEmbedder that never loads BGE-M3
  sample_items      — 3 pre-validated MediaItem objects for quick tests
  scored_candidates — 3 ScoredCandidate objects ready for scoring/explaining
"""
from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from src.schema import EmbeddedItem, MediaItem, ScoredCandidate


# ---------------------------------------------------------------------------
# Async-capable fake anthropic module
# ---------------------------------------------------------------------------

def _build_anthropic_module(response_json: str = '{"reasons":["Great match"],"matched_tags":["mecha"]}'):
    fake_mod = types.ModuleType("anthropic")

    class _Block:
        def __init__(self, text):
            self.text = text

    class _Response:
        def __init__(self, text):
            self.content = [_Block(text)]

    class _SyncMessages:
        def __init__(self, text):
            self._text = text
        def create(self, **kw):
            return _Response(self._text)

    class _AsyncMessages:
        def __init__(self, text):
            self._text = text
        async def create(self, **kw):
            return _Response(self._text)

    class _SyncClient:
        def __init__(self, api_key=None, **kw):
            self.messages = _SyncMessages(response_json)

    class _AsyncClient:
        def __init__(self, api_key=None, **kw):
            self.messages = _AsyncMessages(response_json)

    fake_mod.Anthropic = _SyncClient
    fake_mod.AsyncAnthropic = _AsyncClient
    return fake_mod


# ---------------------------------------------------------------------------
# MockEmbedder — no model loading
# ---------------------------------------------------------------------------

class MockEmbedder:
    """Drop-in Embedder replacement that returns deterministic dummy vectors."""

    DENSE_DIM = 1024

    def embed_dense(self, text: str) -> list[float]:
        h = hash(text) % 1000
        vec = [0.0] * self.DENSE_DIM
        vec[h] = 1.0
        return vec

    def embed_sparse(self, text: str) -> tuple[list[int], list[float]]:
        h = hash(text) % 1000
        return [h, h + 1, h + 2], [0.5, 0.3, 0.2]

    def embed_batch(
        self,
        items: list,
        batch_size: int = 32,
        progress_callback=None,
    ) -> list:
        results = []
        for item in items:
            h = hash(item.dense_text) % 1000
            dense = [0.0] * self.DENSE_DIM
            dense[h] = 1.0
            results.append(
                EmbeddedItem(
                    item=item,
                    dense_vector=dense,
                    sparse_indices=[h, h + 1, h + 2],
                    sparse_values=[0.5, 0.3, 0.2],
                )
            )
        if progress_callback:
            progress_callback(len(items), len(items))
        return results


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    """Settings with a per-test temp SQLite file so stores don't share state."""
    from src.config import Settings
    return Settings(
        anthropic_api_key=None,
        sqlite_path=str(tmp_path / "test_rec.db"),
        lambda_recency=0.05,
        length_scale=12,
    )


@pytest.fixture
def sqlite_store(cfg):
    """A fresh SQLiteStore backed by the per-test temp db."""
    from src.store import SQLiteStore
    store = SQLiteStore(cfg)
    store.create_collection()
    yield store
    store.close()


@pytest.fixture
def anthropic_mock(monkeypatch):
    """Injects a fake anthropic into sys.modules. Returns the module."""
    fake = _build_anthropic_module()
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    return fake


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def sample_items() -> list[MediaItem]:
    """Three minimal but valid MediaItems for quick tests."""
    return [
        MediaItem.model_validate({
            "id": "aaaaaaaa-0000-0000-0000-000000000001",
            "title": "Ghost in the Shell",
            "type": "movie",
            "status": "watched",
            "rating": 9.0,
            "year": 1995,
            "episodes": 1,
            "genres": "Sci-Fi, Action",
            "tags": "cyberpunk, AI, consciousness",
            "review": "A landmark philosophical sci-fi cyberpunk film.",
        }),
        MediaItem.model_validate({
            "id": "aaaaaaaa-0000-0000-0000-000000000002",
            "title": "Akira",
            "type": "movie",
            "status": "plan_to_watch",
            "rating": None,
            "year": 1988,
            "episodes": 1,
            "genres": "Sci-Fi, Action",
            "tags": "cyberpunk, post-apocalyptic, psychic",
        }),
        MediaItem.model_validate({
            "id": "aaaaaaaa-0000-0000-0000-000000000003",
            "title": "The Foundation",
            "type": "book",
            "status": "reading",
            "rating": 8.5,
            "year": 1951,
            "episodes": 255,
            "genres": "Sci-Fi",
            "tags": "galactic-empire, psychohistory, classic",
            "abstract": "Asimov's magnum opus about the fall of a galactic empire.",
        }),
    ]


@pytest.fixture
def scored_candidates(sample_items) -> list[ScoredCandidate]:
    return [
        ScoredCandidate(item=sample_items[0], rrf_score=0.9, dense_score=0.85),
        ScoredCandidate(item=sample_items[1], rrf_score=0.6),
        ScoredCandidate(item=sample_items[2], rrf_score=0.4),
    ]
