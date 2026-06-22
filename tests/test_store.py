"""
Tests for src/store.py — SQLiteStore collection management and upsert.

All tests use real in-memory SQLite via the cfg fixture (tmp_path per test).
"""
from __future__ import annotations


class TestCreateCollection:
    def test_creates_new_table(self, cfg):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        created = store.create_collection()
        assert created is True

    def test_idempotent_on_existing_table(self, cfg):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        created = store.create_collection()
        assert created is False

    def test_collection_info_after_creation(self, cfg):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        info = store.collection_info()
        assert info["points_count"] == 0
        assert info["storage"] == cfg.sqlite_path

    def test_collection_is_queryable_after_creation(self, cfg):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        rows = store.fetch_all()
        assert rows == []


class TestUpsert:
    def test_upsert_stores_items(self, cfg, sample_items, mock_embedder):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        embedded = mock_embedder.embed_batch(sample_items)
        n = store.upsert(embedded)
        assert n == 3

    def test_upsert_reflected_in_count(self, cfg, sample_items, mock_embedder):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        embedded = mock_embedder.embed_batch(sample_items)
        store.upsert(embedded)
        info = store.collection_info()
        assert info["points_count"] == 3

    def test_upsert_payload_contains_required_fields(self, cfg, sample_items, mock_embedder):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        embedded = mock_embedder.embed_batch(sample_items[:1])
        store.upsert(embedded)
        rows = store.fetch_all()
        assert len(rows) == 1
        row = rows[0]
        assert "id" in row
        assert "title" in row
        assert "type" in row
        assert "watch_status" in row
        assert "rating" in row
        assert "year_released" in row
        assert "genres" in row
        assert "tags" in row

    def test_upsert_empty_list_returns_zero(self, cfg):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        assert store.upsert([]) == 0

    def test_upsert_respects_batch_size(self, cfg, sample_items, mock_embedder):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        embedded = mock_embedder.embed_batch(sample_items)
        n = store.upsert(embedded, batch_size=2)
        # All items should still be stored regardless of batch size
        assert n == 3
        assert store.collection_info()["points_count"] == 3

    def test_upsert_is_idempotent(self, cfg, sample_items, mock_embedder):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        embedded = mock_embedder.embed_batch(sample_items[:1])
        store.upsert(embedded)
        store.upsert(embedded)   # same item twice
        assert store.collection_info()["points_count"] == 1


class TestCollectionInfo:
    def test_returns_zero_for_empty_collection(self, cfg):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        info = store.collection_info()
        assert info["points_count"] == 0

    def test_returns_count_after_upsert(self, cfg, sample_items, mock_embedder):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        embedded = mock_embedder.embed_batch(sample_items)
        store.upsert(embedded)
        info = store.collection_info()
        assert info["points_count"] == 3


class TestDelete:
    def test_delete_removes_item(self, cfg, sample_items, mock_embedder):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        embedded = mock_embedder.embed_batch(sample_items[:1])
        store.upsert(embedded)
        item_id = sample_items[0].id
        store.delete(item_id)
        assert store.collection_info()["points_count"] == 0

    def test_delete_nonexistent_is_silent(self, cfg):
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        store.delete("nonexistent-id")   # should not raise
