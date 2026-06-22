"""
Tests for src/retriever.py — HybridRetriever and payload-to-item conversion.

Uses real in-memory SQLite via the cfg fixture.
"""
from __future__ import annotations

from src.core.schema import MediaItem, ParsedQuery, ScoredCandidate


class TestPayloadToItem:
    """Unit-tests for the _payload_to_item helper (no SQLite needed)."""

    def test_basic_mapping(self):
        from src.search.retriever import _payload_to_item
        payload = {
            "id": "test-uuid",
            "title": "Ghost in the Shell",
            "type": "movie",
            "watch_status": "watched",
            "rating": 9.0,
            "year_released": 1995,
            "num_episodes_or_pages": 1,
            "genres": ["Sci-Fi"],
            "tags": ["cyberpunk"],
            "associated_entities": [],
            "local_file_location": None,
            "web_link": "https://example.com",
        }
        item = _payload_to_item(payload)
        assert item.id == "test-uuid"
        assert item.title == "Ghost in the Shell"
        assert item.type == "movie"
        assert item.rating == 9.0
        assert item.year_released == 1995
        assert item.genres == ["Sci-Fi"]
        assert item.web_link == "https://example.com"

    def test_missing_optional_fields_use_none(self):
        from src.search.retriever import _payload_to_item
        item = _payload_to_item({"id": "x", "title": "Minimal"})
        assert item.type is None
        assert item.rating is None
        assert item.year_released is None
        assert item.genres == []

    def test_zero_values_become_none(self):
        from src.search.retriever import _payload_to_item
        item = _payload_to_item({
            "id": "x", "title": "T",
            "rating": 0, "year_released": 0, "num_episodes_or_pages": 0
        })
        assert item.rating is None
        assert item.year_released is None

    def test_entity_list_preserved(self):
        from src.search.retriever import _payload_to_item
        item = _payload_to_item({
            "id": "x", "title": "T",
            "associated_entities": ["Satoshi Kon", "MADHOUSE"]
        })
        assert item.associated_entities == ["Satoshi Kon", "MADHOUSE"]


class TestHybridRetriever:
    """Integration tests using real SQLite via cfg fixture."""

    def _make_store_and_retriever(self, cfg, mock_embedder):
        from src.search.retriever import HybridRetriever
        from src.data.store import SQLiteStore
        store = SQLiteStore(cfg)
        store.create_collection()
        retriever = HybridRetriever(store, mock_embedder, cfg)
        return store, retriever

    def test_empty_semantic_query_uses_scroll(
        self, cfg, mock_embedder, sample_items
    ):
        store, retriever = self._make_store_and_retriever(cfg, mock_embedder)
        embedded = mock_embedder.embed_batch(sample_items)
        store.upsert(embedded)
        parsed = ParsedQuery(semantic_query="")
        candidates = retriever.retrieve(parsed, top_k=10)
        assert isinstance(candidates, list)
        assert all(isinstance(c, ScoredCandidate) for c in candidates)

    def test_semantic_query_returns_scored_candidates(
        self, cfg, mock_embedder, sample_items
    ):
        store, retriever = self._make_store_and_retriever(cfg, mock_embedder)
        embedded = mock_embedder.embed_batch(sample_items)
        store.upsert(embedded)
        parsed = ParsedQuery(semantic_query="cyberpunk philosophical sci-fi")
        candidates = retriever.retrieve(parsed, top_k=3)
        assert isinstance(candidates, list)
        assert all(isinstance(c, ScoredCandidate) for c in candidates)
        assert all(c.rrf_score > 0 for c in candidates)

    def test_results_contain_valid_media_items(
        self, cfg, mock_embedder, sample_items
    ):
        store, retriever = self._make_store_and_retriever(cfg, mock_embedder)
        embedded = mock_embedder.embed_batch(sample_items)
        store.upsert(embedded)
        candidates = retriever.retrieve(ParsedQuery(semantic_query="sci-fi"), top_k=3)
        for c in candidates:
            assert isinstance(c.item, MediaItem)
            assert c.item.title
            assert c.item.id

    def test_top_k_limits_results(
        self, cfg, mock_embedder, sample_items
    ):
        store, retriever = self._make_store_and_retriever(cfg, mock_embedder)
        embedded = mock_embedder.embed_batch(sample_items)
        store.upsert(embedded)
        candidates = retriever.retrieve(ParsedQuery(semantic_query="anything"), top_k=2)
        assert len(candidates) <= 2

    def test_empty_collection_returns_empty_list(self, cfg, mock_embedder):
        _, retriever = self._make_store_and_retriever(cfg, mock_embedder)
        candidates = retriever.retrieve(ParsedQuery(semantic_query="test"), top_k=5)
        assert candidates == []

    def test_filter_only_scroll_on_empty_query(
        self, cfg, mock_embedder, sample_items
    ):
        store, retriever = self._make_store_and_retriever(cfg, mock_embedder)
        embedded = mock_embedder.embed_batch(sample_items)
        store.upsert(embedded)
        parsed = ParsedQuery(semantic_query="")
        candidates = retriever.retrieve(parsed, top_k=10)
        assert len(candidates) == len(sample_items)
        titles = {c.item.title for c in candidates}
        assert "Ghost in the Shell" in titles
