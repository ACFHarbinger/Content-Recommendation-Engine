"""
Phase 8 — Cross-encoder reranker using bge-reranker-v2-m3.

Sits between the HybridRetriever (top-200) and the Scorer to improve
precision before applying business-logic decay functions.

The reranker scores each (query, item_text) pair and re-orders candidates
by cross-encoder score before passing to the Scorer.
"""
from __future__ import annotations

import logging
import time
import threading
from typing import Optional

from .config import Settings, get_settings
from .schema import ScoredCandidate

logger = logging.getLogger(__name__)

_reranker_model = None
_reranker_lock = threading.Lock()


def _get_reranker(model_name: str):
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model
    with _reranker_lock:
        if _reranker_model is not None:
            return _reranker_model
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as e:
            raise RuntimeError(
                "FlagEmbedding not installed. "
                "Run: pip install 'FlagEmbedding>=1.3.5'"
            ) from e
        t0 = time.perf_counter()
        logger.info("Loading reranker %s…", model_name)
        _reranker_model = FlagReranker(model_name, use_fp16=True)
        logger.info("Reranker loaded in %.1fs", time.perf_counter() - t0)
        return _reranker_model


class Reranker:
    """
    Cross-encoder reranker wrapping bge-reranker-v2-m3.

    Usage
    -----
    reranker = Reranker()
    reranked = reranker.rerank(query="mecha anime", candidates=candidates, top_n=20)
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._cfg = settings or get_settings()

    def rerank(
        self,
        query: str,
        candidates: list[ScoredCandidate],
        top_n: Optional[int] = None,
    ) -> list[ScoredCandidate]:
        """
        Re-rank *candidates* using the cross-encoder and return the top *top_n*.

        Each candidate's ``rrf_score`` is replaced with the cross-encoder score
        (normalised to [0, 1] via sigmoid) to feed the downstream Scorer.

        Falls back to the original ordering on any model error.
        """
        if not candidates:
            return candidates
        n = top_n or self._cfg.rerank_top_n

        try:
            reranker = _get_reranker(self._cfg.rerank_model)
        except RuntimeError as exc:
            logger.warning("Reranker unavailable (%s) — skipping.", exc)
            return candidates[:n]

        try:
            import math

            pairs = [(query, c.item.dense_text[:512]) for c in candidates]
            scores = reranker.compute_score(pairs, normalize=True)

            if not isinstance(scores, (list, tuple)):
                scores = [scores]

            reranked = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            return [
                c.model_copy(update={"rrf_score": float(s)})
                for c, s in reranked[:n]
            ]
        except Exception as exc:
            logger.warning("Reranking failed (%s) — using original order.", exc)
            return candidates[:n]
