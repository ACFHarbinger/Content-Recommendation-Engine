"""
Phase 6 — End-to-end RecommendationPipeline.

Wires together all pipeline stages:
    QueryParser → HybridRetriever → [Reranker] → Scorer → Explainer

Each stage is optional and can be disabled via constructor flags, so the
pipeline degrades gracefully when dependencies (Claude API, reranker model)
are unavailable.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from .config import Settings, get_settings
from .embedder import Embedder
from .explainer import Explainer
from .query_parser import QueryParser, _build_qdrant_filter
from .reranker import Reranker
from .retriever import HybridRetriever
from .schema import ExplainedResult, HistoryProfile
from .scorer import Scorer
from .store import QdrantStore

logger = logging.getLogger(__name__)


class RecommendationPipeline:
    """
    Full recommendation pipeline: query → results with explanations.

    Parameters
    ----------
    top_k       : Number of final results to return.
    use_reranker: Whether to apply the cross-encoder reranker (Phase 8).
    explain     : Whether to generate LLM explanations (Phase 5).
    settings    : Override default settings.
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        use_reranker: bool = False,
        explain: bool = True,
        use_history: bool = True,
        settings: Optional[Settings] = None,
    ):
        self._cfg = settings or get_settings()
        self._top_k = top_k or self._cfg.default_top_k
        self._use_reranker = use_reranker
        self._explain = explain
        self._use_history = use_history

        # Shared embedder instance to avoid double model loading
        self._embedder = Embedder(self._cfg.embed_model)

        self._store = QdrantStore(self._cfg)
        self._parser = QueryParser(self._cfg)
        self._retriever = HybridRetriever(self._store, self._embedder, self._cfg)
        self._scorer = Scorer(self._cfg)
        self._explainer = Explainer(self._cfg) if explain else None
        self._reranker = Reranker(self._cfg) if use_reranker else None

    def run(self, query: str) -> list[ExplainedResult]:
        """
        Execute the full pipeline for *query*.

        Returns a list of ExplainedResult sorted by descending
        recommendation_value (rank 1 is best).
        """
        t_start = time.perf_counter()

        # Stage 1 — Parse
        parsed, qdrant_filter = self._parser.parse_to_qdrant(query)
        logger.info(
            "Parsed query: semantic=%r, %d filter(s), length_pref=%s",
            parsed.semantic_query[:60],
            len(parsed.filters),
            parsed.length_preference_episodes,
        )

        # Stage 2 — Retrieve
        fetch_k = self._top_k * 4 if self._use_reranker else self._top_k * 2
        candidates = self._retriever.retrieve(
            parsed, top_k=min(fetch_k, 200), qdrant_filter=qdrant_filter
        )
        logger.info("Retrieved %d candidates.", len(candidates))

        # Stage 3 — Rerank (optional)
        if self._reranker and candidates:
            candidates = self._reranker.rerank(
                query=parsed.semantic_query or query,
                candidates=candidates,
                top_n=self._top_k * 2,
            )
            logger.info("Reranked to %d candidates.", len(candidates))

        # Stage 4 — Score (with optional watch-history boost)
        history = self._build_history_profile() if self._use_history else None
        ranked = self._scorer.score(candidates[: self._top_k * 2], parsed, history)
        ranked = ranked[: self._top_k]
        logger.info(
            "Scored %d results; top recommendation_value=%.4f",
            len(ranked),
            ranked[0].recommendation_value if ranked else 0.0,
        )

        # Stage 5 — Explain
        if self._explainer and ranked:
            results = self._explainer.explain_batch(ranked, query)
        else:
            from .schema import ExplainedResult as ER
            results = [ER(**r.model_dump(), reasons=[], matched_tags=[]) for r in ranked]

        elapsed = time.perf_counter() - t_start
        logger.info("Pipeline completed in %.2fs", elapsed)
        return results

    # ------------------------------------------------------------------
    # Watch-history profile
    # ------------------------------------------------------------------

    def _build_history_profile(self) -> Optional[HistoryProfile]:
        """
        Scroll the Qdrant collection for watched/read items rated above
        ``history_min_rating`` and aggregate their tags + genres into a
        HistoryProfile for the downstream Scorer.

        Returns None on any error so the pipeline continues without boosting.
        """
        try:
            client = self._store._get_client()
            min_rating = self._cfg.history_min_rating

            # Build a payload filter for watched/read + highly rated
            try:
                from qdrant_client.models import (
                    FieldCondition, Filter, MatchAny, Range,
                )
                filt = Filter(
                    must=[
                        FieldCondition(
                            key="watch_status",
                            match=MatchAny(any=["watched", "reading"]),
                        ),
                        FieldCondition(
                            key="rating",
                            range=Range(gte=min_rating),
                        ),
                    ]
                )
            except Exception:
                filt = None

            points, _ = client.scroll(
                collection_name=self._cfg.qdrant_collection,
                scroll_filter=filt,
                limit=500,
                with_payload=True,
                with_vectors=False,
            )
            payloads = [p.payload for p in points if p.payload]
            profile = HistoryProfile.from_payloads(payloads)
            logger.debug(
                "History profile: %d items, %d tags, %d genres",
                profile.item_count,
                len(profile.preferred_tags),
                len(profile.preferred_genres),
            )
            return profile
        except Exception as exc:
            logger.debug("Could not build history profile: %s", exc)
            return None
