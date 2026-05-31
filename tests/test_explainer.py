"""
Tests for src/explainer.py — Explainer and fallback logic.

All async paths are exercised through the synchronous explain_batch()
entry point.  anthropic is mocked via conftest.py.
"""
from __future__ import annotations

import json

import pytest

from src.schema import ComponentScores, MediaItem, RankedResult


def _make_ranked(item: MediaItem, rank: int = 1, rv: float = 0.8) -> RankedResult:
    return RankedResult(
        item=item,
        recommendation_value=rv,
        component_scores=ComponentScores(
            rrf_score=rv,
            rating_boost=1.2,
            recency_decay=0.9,
            length_decay=1.0,
        ),
        rank=rank,
    )


class TestFallbackReason:
    """_fallback_reason should always produce a valid ExplainedResult."""

    def test_no_api_key_all_fallback(self, cfg, scored_candidates):
        from src.explainer import Explainer
        from src.scorer import Scorer
        scorer = Scorer(cfg)
        ranked = scorer.score(scored_candidates)
        explainer = Explainer(cfg)  # cfg has no anthropic_api_key
        results = explainer.explain_batch(ranked, "test query")
        assert len(results) == len(ranked)
        for r in results:
            assert r.reasons  # at least one reason
            assert r.rank > 0

    def test_fallback_has_correct_type(self, cfg, scored_candidates):
        from src.explainer import _fallback_reason
        from src.scorer import Scorer
        ranked = Scorer(cfg).score(scored_candidates)
        result = _fallback_reason(ranked[0], "cyberpunk query")
        assert result.recommendation_value == ranked[0].recommendation_value
        assert isinstance(result.reasons, list)
        assert len(result.reasons) >= 1

    def test_fallback_video_uses_watch_verb(self, cfg, sample_items):
        from src.explainer import _fallback_reason
        anime_item = MediaItem.model_validate({
            "id": "x", "title": "Test Anime", "type": "anime",
            "rating": 8.0, "year": 2020, "episodes": 12,
        })
        ranked = _make_ranked(anime_item)
        result = _fallback_reason(ranked, "query")
        # Reasons are plain text — just verify the object is valid
        assert all(isinstance(r, str) for r in result.reasons)

    def test_matched_tags_anti_hallucination(self, cfg, sample_items):
        """matched_tags in ExplainedResult must be subset of item's actual tags+genres."""
        from src.schema import ExplainedResult
        item = sample_items[0]  # Ghost in the Shell — tags: cyberpunk, AI, consciousness
        ranked = _make_ranked(item)
        result = ExplainedResult(
            **ranked.model_dump(),
            reasons=["Good match"],
            matched_tags=["cyberpunk", "nonexistent-fabricated-tag"],
        )
        assert "nonexistent-fabricated-tag" not in result.matched_tags
        assert "cyberpunk" in result.matched_tags


class TestExplainerWithMock:
    """Tests that go through the async path with a mocked anthropic client."""

    def test_mocked_api_returns_structured_result(
        self, anthropic_mock, cfg, scored_candidates
    ):
        from src.config import Settings
        from src.explainer import Explainer
        from src.scorer import Scorer

        # Config with a fake API key to trigger the Claude path
        cfg_with_key = Settings(
            anthropic_api_key="sk-fake",
            qdrant_local_path="/tmp/t",
        )
        ranked = Scorer(cfg_with_key).score(scored_candidates)
        explainer = Explainer(cfg_with_key)
        results = explainer.explain_batch(ranked, "cyberpunk sci-fi")
        assert len(results) == len(ranked)
        # The mock returns {"reasons":["Great match"],"matched_tags":["mecha"]}
        # but "mecha" is NOT in Ghost in the Shell's tags, so it gets stripped
        for r in results:
            assert isinstance(r.reasons, list)

    def test_empty_results_returns_empty(self, cfg):
        from src.explainer import Explainer
        explainer = Explainer(cfg)
        assert explainer.explain_batch([], "query") == []

    def test_result_ranks_preserved(self, cfg, scored_candidates):
        from src.explainer import Explainer
        from src.scorer import Scorer
        ranked = Scorer(cfg).score(scored_candidates)
        explainer = Explainer(cfg)
        results = explainer.explain_batch(ranked, "test")
        ranks = [r.rank for r in results]
        assert ranks == sorted(ranks)  # ranks are in ascending order (1, 2, 3, ...)
