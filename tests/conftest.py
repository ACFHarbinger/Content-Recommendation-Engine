"""
Shared test fixtures for the Recommendation Engine test suite.

Provides:
  qdrant_mock     — injects a stateful in-memory FakeQdrantClient into sys.modules
  anthropic_mock  — injects a minimal async-capable fake anthropic module
  mock_embedder   — returns a MockEmbedder that never loads BGE-M3
  sample_items    — 3 pre-validated MediaItem objects for quick tests
  scored_candidates — 3 ScoredCandidate objects ready for scoring/explaining
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from src.schema import EmbeddedItem, MediaItem, ScoredCandidate

# ---------------------------------------------------------------------------
# In-memory Qdrant client + model stubs
# ---------------------------------------------------------------------------

class _FakeBase:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeCollectionEntry:
    def __init__(self, name: str):
        self.name = name


class _FakeCollectionsList:
    def __init__(self, names: list[str]):
        self.collections = [_FakeCollectionEntry(n) for n in names]


class _FakeCollectionInfo:
    def __init__(self, points_count: int):
        self.points_count = points_count


class _FakePointStruct:
    def __init__(self, id, vector, payload=None):
        self.id = id
        self.vector = vector
        self.payload = payload or {}


class _FakeScoredPoint:
    def __init__(self, payload: dict, score: float):
        self.id = payload.get("id", "x")
        self.payload = payload
        self.score = score


class _FakeQueryResponse:
    def __init__(self, points: list):
        self.points = points


class _FakeQdrantCollection:
    def __init__(self):
        self.points: dict[str, _FakePointStruct] = {}


class FakeQdrantClient:
    """
    Stateful in-memory Qdrant client substitute.

    Records all calls and stores upserted points so retrieval tests
    can assert on what was written.  Search/query methods return
    stored points in insertion order (no actual ANN — correctness
    of ranking is tested at the unit level, not here).
    """

    def __init__(self, path=None, url=None, api_key=None):
        self._colls: dict[str, _FakeQdrantCollection] = {}
        self.calls: list[tuple[str, Any]] = []   # [(method_name, kwargs), ...]

    # ---- collection management ----

    def get_collections(self):
        return _FakeCollectionsList(list(self._colls.keys()))

    def create_collection(self, collection_name, vectors_config=None,
                          sparse_vectors_config=None, **_kw):
        self.calls.append(("create_collection", collection_name))
        self._colls.setdefault(collection_name, _FakeQdrantCollection())

    def create_payload_index(self, collection_name, field_name, field_schema=None, **_kw):
        self.calls.append(("create_payload_index", (collection_name, field_name)))

    def delete_collection(self, collection_name):
        self._colls.pop(collection_name, None)

    def get_collection(self, collection_name):
        col = self._colls.get(collection_name)
        return _FakeCollectionInfo(len(col.points) if col else 0)

    # ---- upsert / delete ----

    def upsert(self, collection_name, points, **_kw):
        col = self._colls.setdefault(collection_name, _FakeQdrantCollection())
        for p in points:
            col.points[str(p.id)] = p
        self.calls.append(("upsert", (collection_name, len(points))))

    def delete(self, collection_name, points_selector=None, **_kw):
        col = self._colls.get(collection_name)
        if col and points_selector is not None:
            for pid in getattr(points_selector, "points", []):
                col.points.pop(str(pid), None)

    # ---- search / scroll / query ----

    def scroll(self, collection_name, scroll_filter=None, limit=100,
               with_payload=True, with_vectors=False, offset=None, **_kw):
        col = self._colls.get(collection_name)
        if not col:
            return [], None
        pts = list(col.points.values())[:limit]
        return [_FakeScoredPoint(p.payload, 1.0) for p in pts], None

    def search(self, collection_name, query_vector=None, query_filter=None,
               limit=10, with_payload=True, **_kw):
        col = self._colls.get(collection_name)
        if not col:
            return []
        pts = list(col.points.values())[:limit]
        return [_FakeScoredPoint(p.payload, 0.9 - i * 0.05) for i, p in enumerate(pts)]

    def query_points(self, collection_name, prefetch=None, query=None,
                     limit=10, with_payload=True, **_kw):
        col = self._colls.get(collection_name)
        if not col:
            return _FakeQueryResponse([])
        pts = list(col.points.values())[:limit]
        return _FakeQueryResponse(
            [_FakeScoredPoint(p.payload, 0.9 - i * 0.05) for i, p in enumerate(pts)]
        )


def _build_qdrant_module(client_instance: FakeQdrantClient):
    """Build a fake qdrant_client module tree around a given client instance."""
    qdrant_mod = types.ModuleType("qdrant_client")
    models_mod = types.ModuleType("qdrant_client.models")

    # Stick all needed stubs onto models_mod
    for name, cls in [
        ("Filter", _FakeBase),
        ("FieldCondition", _FakeBase),
        ("MatchValue", _FakeBase),
        ("MatchAny", _FakeBase),
        ("MatchExcept", _FakeBase),
        ("Range", _FakeBase),
        ("PointStruct", _FakePointStruct),
        ("SparseVector", _FakeBase),
        ("NamedVector", _FakeBase),
        ("NamedSparseVector", _FakeBase),
        ("PointIdsList", _FakeBase),
        ("VectorParams", _FakeBase),
        ("SparseVectorParams", _FakeBase),
        ("SparseIndexParams", _FakeBase),
        ("Prefetch", _FakeBase),
        ("FusionQuery", _FakeBase),
    ]:
        setattr(models_mod, name, cls)

    # Enum-like stubs
    class _Distance:
        COSINE = "cosine"; DOT = "dot"

    class _Fusion:
        RRF = "rrf"; DBSF = "dbsf"

    class _PayloadSchemaType:
        KEYWORD = "keyword"; FLOAT = "float"; INTEGER = "integer"

    models_mod.Distance = _Distance
    models_mod.Fusion = _Fusion
    models_mod.PayloadSchemaType = _PayloadSchemaType

    qdrant_mod.models = models_mod
    qdrant_mod.QdrantClient = lambda **kw: client_instance

    return qdrant_mod, models_mod


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
        # Use hash of text so different texts get different vectors
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
def fake_client() -> FakeQdrantClient:
    """A fresh in-memory FakeQdrantClient per test."""
    return FakeQdrantClient()


@pytest.fixture
def qdrant_mock(monkeypatch, fake_client):
    """
    Injects a fake qdrant_client into sys.modules for one test.
    Returns the underlying FakeQdrantClient so tests can inspect state.
    """
    qdrant_mod, models_mod = _build_qdrant_module(fake_client)
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_mod)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models_mod)
    return fake_client


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


@pytest.fixture
def cfg():
    """A minimal Settings object with no external dependencies."""
    from src.config import Settings
    return Settings(
        anthropic_api_key=None,
        qdrant_local_path="/tmp/test_rec_engine",
        lambda_recency=0.05,
        length_scale=12,
    )
