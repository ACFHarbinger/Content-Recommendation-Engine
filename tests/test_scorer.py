"""
Phase 4 tests — Scorer (Recommendation Value computation).

Tests cover all three decay/boost functions independently, their multiplicative
combination, and edge cases (missing rating, missing year, no length preference).
"""
from __future__ import annotations

import math

import pytest

from src.config import Settings
from src.schema import MediaItem, ParsedQuery, ScoredCandidate
from src.scorer import Scorer, _CURRENT_YEAR


def _cfg(**kwargs) -> Settings:
    defaults = {"qdrant_local_path": "/tmp/test", "lambda_recency": 0.05}
    defaults.update(kwargs)
    return Settings(**defaults)


def _candidate(
    rating: float | None = 8.0,
    year: int | None = 2020,
    episodes: int | None = 12,
    rrf: float = 0.5,
    title: str = "Test",
) -> ScoredCandidate:
    item = MediaItem.model_validate(
        {
            "id": "test-id",
            "title": title,
            "rating": rating,
            "year": year,
            "episodes": episodes,
        }
    )
    return ScoredCandidate(item=item, rrf_score=rrf)


class TestRatingBoost:
    def test_no_rating_is_neutral(self):
        scorer = Scorer(_cfg())
        assert scorer._rating_boost(None) == 1.0
        assert scorer._rating_boost(0.0) == 1.0

    def test_perfect_rating_boosts(self):
        scorer = Scorer(_cfg())
        boost = scorer._rating_boost(10.0)
        assert boost > 1.5  # 1 + log(2) ≈ 1.693

    def test_half_rating(self):
        scorer = Scorer(_cfg())
        boost = scorer._rating_boost(5.0)
        assert 1.3 < boost < 1.5

    def test_monotonically_increasing(self):
        scorer = Scorer(_cfg())
        boosts = [scorer._rating_boost(r) for r in [0, 3, 5, 7, 10]]
        assert boosts == sorted(boosts)


class TestRecencyDecay:
    def test_current_year_no_decay(self):
        scorer = Scorer(_cfg(lambda_recency=0.05))
        assert scorer._recency_decay(_CURRENT_YEAR) == pytest.approx(1.0)

    def test_old_item_decays(self):
        scorer = Scorer(_cfg(lambda_recency=0.05))
        old = scorer._recency_decay(1995)
        assert old < 0.5  # older than 20 years at λ=0.05 decays significantly

    def test_zero_lambda_no_decay(self):
        scorer = Scorer(_cfg(lambda_recency=0.0))
        assert scorer._recency_decay(1900) == pytest.approx(1.0)

    def test_unknown_year_is_neutral(self):
        scorer = Scorer(_cfg())
        assert scorer._recency_decay(None) == 1.0
        assert scorer._recency_decay(0) == 1.0

    def test_monotonically_decreasing_with_age(self):
        scorer = Scorer(_cfg(lambda_recency=0.05))
        decays = [scorer._recency_decay(y) for y in [_CURRENT_YEAR, 2010, 2000, 1990]]
        assert decays == sorted(decays, reverse=True)


class TestLengthDecay:
    def test_no_preference_is_neutral(self):
        scorer = Scorer(_cfg())
        assert scorer._length_decay(50, None) == 1.0

    def test_exact_match_is_max(self):
        scorer = Scorer(_cfg(length_scale=12))
        assert scorer._length_decay(26, 26) == pytest.approx(1.0)

    def test_deviation_penalises(self):
        scorer = Scorer(_cfg(length_scale=12))
        penalty = scorer._length_decay(100, 12)
        assert penalty < 0.5

    def test_unknown_episodes_is_neutral(self):
        scorer = Scorer(_cfg())
        assert scorer._length_decay(None, 26) == 1.0


class TestScorerIntegration:
    def test_high_rated_recent_outranks_low_rated_old(self):
        scorer = Scorer(_cfg(lambda_recency=0.05))
        good = _candidate(rating=9.5, year=_CURRENT_YEAR, rrf=0.5)
        bad = _candidate(rating=5.0, year=1980, rrf=0.5, title="Old")
        results = scorer.score([good, bad])
        assert results[0].item.rating == 9.5

    def test_ranking_is_sorted_descending(self):
        scorer = Scorer(_cfg())
        candidates = [_candidate(rrf=r) for r in [0.3, 0.8, 0.5, 0.1]]
        results = scorer.score(candidates)
        vals = [r.recommendation_value for r in results]
        assert vals == sorted(vals, reverse=True)

    def test_rank_is_assigned_from_one(self):
        scorer = Scorer(_cfg())
        results = scorer.score([_candidate(), _candidate(rrf=0.9)])
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_length_preference_flows_from_parsed_query(self):
        scorer = Scorer(_cfg(length_scale=12))
        long_show = _candidate(episodes=200, rrf=0.7, title="Long")
        short_show = _candidate(episodes=12, rrf=0.7, title="Short")
        pq = ParsedQuery(length_preference_episodes=12)
        results = scorer.score([long_show, short_show], pq)
        # Short show should win because it matches length preference
        assert results[0].item.title == "Short"

    def test_zero_rrf_stays_near_zero(self):
        scorer = Scorer(_cfg())
        c = _candidate(rating=10.0, year=_CURRENT_YEAR, rrf=0.0)
        results = scorer.score([c])
        assert results[0].recommendation_value == pytest.approx(0.0)

    def test_component_scores_present(self):
        scorer = Scorer(_cfg())
        results = scorer.score([_candidate()])
        cs = results[0].component_scores
        assert cs.rrf_score > 0
        assert cs.rating_boost >= 1.0
        assert 0.0 < cs.recency_decay <= 1.0
        assert cs.length_decay == 1.0  # no length preference


class TestHistoryBoost:
    def test_no_profile_is_neutral(self):
        scorer = Scorer(_cfg())
        c = _candidate(rrf=0.5)
        assert scorer._history_boost(c.item, None) == 1.0

    def test_empty_profile_is_neutral(self):
        from src.schema import HistoryProfile
        scorer = Scorer(_cfg())
        profile = HistoryProfile()
        assert scorer._history_boost(_candidate().item, profile) == 1.0

    def test_full_overlap_boosts_by_weight(self):
        from src.schema import HistoryProfile, MediaItem
        scorer = Scorer(_cfg(history_boost_weight=0.2))
        item = MediaItem.model_validate({
            "id": "x", "title": "T",
            "tags": "cyberpunk, AI", "genres": "Sci-Fi",
        })
        profile = HistoryProfile(
            preferred_tags=frozenset({"cyberpunk", "ai"}),
            preferred_genres=frozenset({"sci-fi"}),
            item_count=5,
        )
        boost = scorer._history_boost(item, profile)
        # All three signals overlap → full overlap → 1 + 0.2 * 1.0 = 1.2
        assert boost == pytest.approx(1.2)

    def test_partial_overlap_between_neutral_and_max(self):
        from src.schema import HistoryProfile, MediaItem
        scorer = Scorer(_cfg(history_boost_weight=0.2))
        item = MediaItem.model_validate({
            "id": "x", "title": "T",
            "tags": "cyberpunk, romance",
            "genres": "Sci-Fi",
        })
        profile = HistoryProfile(
            preferred_tags=frozenset({"cyberpunk"}),
            preferred_genres=frozenset({"sci-fi"}),
            item_count=3,
        )
        boost = scorer._history_boost(item, profile)
        assert 1.0 < boost < 1.2  # partial overlap

    def test_history_boost_applied_in_score(self):
        from src.schema import HistoryProfile, MediaItem
        scorer = Scorer(_cfg(history_boost_weight=0.5))
        item = MediaItem.model_validate({
            "id": "x", "title": "T",
            "tags": "cyberpunk", "genres": "Sci-Fi",
            "rating": 8.0, "year": 2020, "episodes": 12,
        })
        profile = HistoryProfile(
            preferred_tags=frozenset({"cyberpunk"}),
            preferred_genres=frozenset({"sci-fi"}),
            item_count=2,
        )
        no_hist = scorer.score([ScoredCandidate(item=item, rrf_score=0.5)])
        with_hist = scorer.score([ScoredCandidate(item=item, rrf_score=0.5)], history_profile=profile)
        assert with_hist[0].recommendation_value > no_hist[0].recommendation_value
