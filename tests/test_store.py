"""
Tests for src/store.py — QdrantStore collection management and upsert.

All tests use the qdrant_mock fixture from conftest.py (no real Qdrant).
"""
from __future__ import annotations

import pytest

from src.schema import EmbeddedItem


class TestCreateCollection:
    def test_creates_new_collection(self, qdrant_mock, cfg):
        from src.store import QdrantStore
        store = QdrantStore(cfg)
        created = store.create_collection()
        assert created is True
        assert "create_collection" in [c[0] for c in qdrant_mock.calls]

    def test_idempotent_on_existing_collection(self, qdrant_mock, cfg):
        from src.store import QdrantStore
        store = QdrantStore(cfg)
        store.create_collection()
        # Second call should return False (already exists)
        created = store.create_collection()
        assert created is False

    def test_creates_payload_indexes(self, qdrant_mock, cfg):
        from src.store import QdrantStore
        store = QdrantStore(cfg)
        store.create_collection()
        index_calls = [c for c in qdrant_mock.calls if c[0] == "create_payload_index"]
        indexed_fields = {c[1][1] for c in index_calls}
        assert "type" in indexed_fields
        assert "watch_status" in indexed_fields
        assert "rating" in indexed_fields
        assert "year_released" in indexed_fields


class TestUpsert:
    def test_upsert_stores_items(self, qdrant_mock, cfg, sample_items, mock_embedder):
        from src.store import QdrantStore
        store = QdrantStore(cfg)
        store.create_collection()
        embedded = mock_embedder.embed_batch(sample_items)
        n = store.upsert(embedded)
        assert n == 3

    def test_upsert_calls_qdrant_upsert(self, qdrant_mock, cfg, sample_items, mock_embedder):
        from src.store import QdrantStore
        store = QdrantStore(cfg)
        store.create_collection()
        embedded = mock_embedder.embed_batch(sample_items)
        store.upsert(embedded)
        upsert_calls = [c for c in qdrant_mock.calls if c[0] == "upsert"]
        assert upsert_calls, "upsert should have been called"
        total_upserted = sum(c[1][1] for c in upsert_calls)
        assert total_upserted == 3

    def test_upsert_payload_contains_required_fields(self, qdrant_mock, cfg, sample_items, mock_embedder):
        from src.store import QdrantStore
        store = QdrantStore(cfg)
        store.create_collection()
        embedded = mock_embedder.embed_batch(sample_items[:1])
        store.upsert(embedded)
        # Inspect stored point payload
        col = qdrant_mock._colls.get(cfg.qdrant_collection)
        assert col is not None
        point = next(iter(col.points.values()))
        payload = point.payload
        assert "id" in payload
        assert "title" in payload
        assert "type" in payload
        assert "watch_status" in payload
        assert "rating" in payload
        assert "year_released" in payload
        assert "genres" in payload
        assert "tags" in payload

    def test_upsert_empty_list_returns_zero(self, qdrant_mock, cfg):
        from src.store import QdrantStore
        store = QdrantStore(cfg)
        store.create_collection()
        assert store.upsert([]) == 0

    def test_upsert_respects_batch_size(self, qdrant_mock, cfg, sample_items, mock_embedder):
        from src.store import QdrantStore
        store = QdrantStore(cfg)
        store.create_collection()
        embedded = mock_embedder.embed_batch(sample_items)
        store.upsert(embedded, batch_size=2)
        upsert_calls = [c for c in qdrant_mock.calls if c[0] == "upsert"]
        assert len(upsert_calls) == 2  # 2 + 1 items in two batches


class TestCollectionInfo:
    def test_returns_zero_for_empty_collection(self, qdrant_mock, cfg):
        from src.store import QdrantStore
        store = QdrantStore(cfg)
        store.create_collection()
        info = store.collection_info()
        assert info["points_count"] == 0
        assert info["collection"] == cfg.qdrant_collection

    def test_returns_count_after_upsert(self, qdrant_mock, cfg, sample_items, mock_embedder):
        from src.store import QdrantStore
        store = QdrantStore(cfg)
        store.create_collection()
        embedded = mock_embedder.embed_batch(sample_items)
        store.upsert(embedded)
        info = store.collection_info()
        assert info["points_count"] == 3


class TestDelete:
    def test_delete_removes_point(self, qdrant_mock, cfg, sample_items, mock_embedder):
        from src.store import QdrantStore
        store = QdrantStore(cfg)
        store.create_collection()
        embedded = mock_embedder.embed_batch(sample_items[:1])
        store.upsert(embedded)
        item_id = sample_items[0].id
        store.delete(item_id)
        col = qdrant_mock._colls.get(cfg.qdrant_collection)
        assert str(item_id) not in col.points
