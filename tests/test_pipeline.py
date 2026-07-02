"""
Tests for src/pipeline.py — RecommendationPipeline end-to-end orchestration.

All external dependencies (BGE-M3, Anthropic) are replaced with
the mock fixtures from conftest.py.  SQLite uses a real temp-file db.
"""

from __future__ import annotations

from src.core.schema import ExplainedResult


def _make_pipeline(cfg, mock_embedder, explain=False, use_reranker=False):
    """Wire up a pipeline with all mocked dependencies."""
    from src.search.explainer import Explainer
    from src.search.pipeline import RecommendationPipeline
    from src.search.query_parser import QueryParser
    from src.search.retriever import HybridRetriever
    from src.search.scorer import Scorer
    from src.data.store import SQLiteStore

    store = SQLiteStore(cfg)
    store.create_collection()

    pipeline = RecommendationPipeline.__new__(RecommendationPipeline)
    pipeline._cfg = cfg
    pipeline._top_k = cfg.default_top_k
    pipeline._use_reranker = use_reranker
    pipeline._explain = explain
    pipeline._embedder = mock_embedder
    pipeline._parser = QueryParser(cfg)
    pipeline._retriever = HybridRetriever(store, mock_embedder, cfg)
    pipeline._scorer = Scorer(cfg)
    pipeline._explainer = Explainer(cfg) if explain else None
    pipeline._reranker = None  # skip reranker in unit tests
    pipeline._use_history = False  # disable history profile in unit tests
    pipeline._store = store
    return pipeline


def _populate(cfg, mock_embedder, items):
    from src.data.store import SQLiteStore

    store = SQLiteStore(cfg)
    store.create_collection()
    embedded = mock_embedder.embed_batch(items)
    store.upsert(embedded)


class TestPipelineRun:
    def test_returns_explained_results(self, cfg, mock_embedder, sample_items):
        _populate(cfg, mock_embedder, sample_items)
        pipeline = _make_pipeline(cfg, mock_embedder)
        results = pipeline.run("cyberpunk sci-fi")
        assert isinstance(results, list)
        assert all(isinstance(r, ExplainedResult) for r in results)

    def test_results_sorted_by_recommendation_value(
        self, cfg, mock_embedder, sample_items
    ):
        _populate(cfg, mock_embedder, sample_items)
        pipeline = _make_pipeline(cfg, mock_embedder)
        results = pipeline.run("any query")
        rvs = [r.recommendation_value for r in results]
        assert rvs == sorted(rvs, reverse=True)

    def test_ranks_start_at_one(self, cfg, mock_embedder, sample_items):
        _populate(cfg, mock_embedder, sample_items)
        pipeline = _make_pipeline(cfg, mock_embedder)
        results = pipeline.run("any query")
        if results:
            assert results[0].rank == 1

    def test_top_k_limits_output(self, cfg, mock_embedder, sample_items):
        _populate(cfg, mock_embedder, sample_items)
        pipeline = _make_pipeline(cfg, mock_embedder)
        pipeline._top_k = 2
        results = pipeline.run("anything")
        assert len(results) <= 2

    def test_no_explain_produces_empty_reasons(self, cfg, mock_embedder, sample_items):
        _populate(cfg, mock_embedder, sample_items)
        pipeline = _make_pipeline(cfg, mock_embedder, explain=False)
        results = pipeline.run("any query")
        for r in results:
            assert r.reasons == []
            assert r.matched_tags == []

    def test_empty_collection_returns_empty_list(self, cfg, mock_embedder):
        pipeline = _make_pipeline(cfg, mock_embedder)
        results = pipeline.run("any query")
        assert results == []

    def test_results_contain_valid_items(self, cfg, mock_embedder, sample_items):
        _populate(cfg, mock_embedder, sample_items)
        pipeline = _make_pipeline(cfg, mock_embedder)
        results = pipeline.run("sci-fi")
        for r in results:
            assert r.item.id
            assert r.item.title

    def test_component_scores_present(self, cfg, mock_embedder, sample_items):
        _populate(cfg, mock_embedder, sample_items)
        pipeline = _make_pipeline(cfg, mock_embedder)
        results = pipeline.run("any")
        for r in results:
            cs = r.component_scores
            assert cs.rrf_score >= 0
            assert cs.rating_boost >= 1.0
            assert 0.0 < cs.recency_decay <= 1.0
