"""
Phase 4 — Recommendation Value scorer.

Applies four multiplicative factors to the raw RRF score:

    RecommendationValue =
        rrf_score × rating_boost × recency_decay × length_decay × history_boost

rating_boost   = 1 + log(1 + rating/10)
recency_decay  = exp(-λ × (current_year - year))
length_decay   = Gaussian, only when length_preference_episodes is set
history_boost  = 1 + weight × overlap_fraction(item, history_profile)
                 (Phase 9: only when a HistoryProfile is supplied)

All factors are multiplicative so a zero-relevance item (rrf_score ≈ 0)
always scores near 0 regardless of quality, recency, or history.
"""

from __future__ import annotations

import datetime
import logging
import math
from typing import Optional

from src.core.config import Settings, get_settings  # pyrefly: ignore [missing-import]
from src.core.schema import (  # pyrefly: ignore [missing-import]
    ComponentScores,
    HistoryProfile,
    ParsedQuery,
    RankedResult,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)

_CURRENT_YEAR = datetime.date.today().year


class Scorer:
    """Transform raw retrieval scores into a final Recommendation Value."""

    def __init__(self, settings: Optional[Settings] = None):
        self._cfg = settings or get_settings()

    def score(
        self,
        candidates: list[ScoredCandidate],
        parsed_query: Optional[ParsedQuery] = None,
        history_profile: Optional[HistoryProfile] = None,
    ) -> list[RankedResult]:
        """
        Rank *candidates* by RecommendationValue.

        Args:
            candidates:       Output from HybridRetriever.
            parsed_query:     Used to extract ``length_preference_episodes``.
            history_profile:  Optional watch-history signal for Phase 9 boost.
        """
        length_pref = parsed_query.length_preference_episodes if parsed_query else None
        results = [self._rank_one(c, length_pref, history_profile) for c in candidates]
        results.sort(key=lambda r: r.recommendation_value, reverse=True)
        for i, r in enumerate(results):
            results[i] = r.model_copy(update={"rank": i + 1})
        return results

    # ------------------------------------------------------------------
    # Per-candidate helpers
    # ------------------------------------------------------------------

    def _rank_one(
        self,
        candidate: ScoredCandidate,
        length_preference: Optional[int],
        history_profile: Optional[HistoryProfile],
    ) -> RankedResult:
        item = candidate.item
        rrf = candidate.rrf_score

        rb = self._rating_boost(item.rating)
        rd = self._recency_decay(item.year_released)
        ld = self._length_decay(item.num_episodes_or_pages, length_preference)
        hb = self._history_boost(item, history_profile)

        value = rrf * rb * rd * ld * hb

        return RankedResult(
            item=item,
            recommendation_value=round(value, 6),
            component_scores=ComponentScores(
                rrf_score=round(rrf, 6),
                rating_boost=round(rb, 4),
                recency_decay=round(rd, 4),
                length_decay=round(ld, 4),
            ),
            rank=0,
        )

    def _rating_boost(self, rating: Optional[float]) -> float:
        """1 + log(1 + normalised_rating).  No rating → 1.0 (neutral)."""
        if rating is None or rating <= 0:
            return 1.0
        normalised = min(rating, 10.0) / 10.0
        return 1.0 + math.log(1.0 + normalised)

    def _recency_decay(self, year: Optional[int]) -> float:
        """exp(-λ × age_in_years).  Unknown year → 1.0 (no penalty)."""
        if year is None or year <= 0:
            return 1.0
        age = max(0, _CURRENT_YEAR - year)
        return math.exp(-self._cfg.lambda_recency * age)

    def _length_decay(
        self,
        episodes: Optional[int],
        preference: Optional[int],
    ) -> float:
        """Gaussian decay around the preferred episode count.
        When no preference is specified the decay is 1.0 (no penalty)."""
        if preference is None or episodes is None or episodes <= 0:
            return 1.0
        origin = preference
        scale = max(self._cfg.length_scale, 1)
        return math.exp(-0.5 * ((episodes - origin) / scale) ** 2)

    def _history_boost(
        self,
        item,
        profile: Optional[HistoryProfile],
    ) -> float:
        """
        Mild multiplicative boost for items whose tags/genres overlap with the
        user's proven preferences (derived from their highly-rated watched items).

        boost = 1 + weight × overlap_fraction(item, profile)

        Returns 1.0 (neutral) when no history profile is supplied.
        """
        if profile is None or profile.item_count == 0:
            return 1.0
        overlap = profile.overlap_fraction(item)
        return 1.0 + self._cfg.history_boost_weight * overlap
