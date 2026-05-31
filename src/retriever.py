"""
Phase 3 — Hybrid Retrieval: dense k-NN + sparse k-NN fused with RRF (or DBSF).

Uses Qdrant's native query API with Prefetch to run both search legs in a
single round trip, then applies score fusion server-side.

Falls back to dense-only search when the sparse index has no data (e.g. a
freshly created collection where sparse encoding failed).
"""
from __future__ import annotations

import logging
from typing import Optional

from .config import Settings, get_settings
from .embedder import Embedder
from .schema import MediaItem, ParsedQuery, ScoredCandidate
from .store import QdrantStore

logger = logging.getLogger(__name__)


def _payload_to_item(payload: dict) -> MediaItem:
    """Convert a Qdrant point payload back into a MediaItem."""
    return MediaItem.model_validate(
        {
            "id": payload.get("id", ""),
            "title": payload.get("title", ""),
            "type": payload.get("type") or None,
            "status": payload.get("watch_status") or None,
            "rating": payload.get("rating") or None,
            "year": payload.get("year_released") or None,
            "episodes": payload.get("num_episodes_or_pages") or None,
            "genres": payload.get("genres", []),
            "tags": payload.get("tags", []),
            "associated_entities": payload.get("associated_entities", []),
            "local_file": payload.get("local_file_location") or None,
            "web_link": payload.get("web_link") or None,
        }
    )


class HybridRetriever:
    """
    Retrieves top-K candidates from Qdrant using hybrid dense+sparse search.

    Strategy:
      1. Embed semantic_query → dense vector + sparse vector.
      2. Issue a Prefetch query with both vectors, filtered by ParsedQuery.filters.
      3. Fuse with RRF (default) or DBSF.
      4. Return ScoredCandidate list with per-leg scores where available.

    Falls back to dense-only on import errors or missing sparse data.
    """

    def __init__(
        self,
        store: Optional[QdrantStore] = None,
        embedder: Optional[Embedder] = None,
        settings: Optional[Settings] = None,
    ):
        self._cfg = settings or get_settings()
        self._store = store or QdrantStore(self._cfg)
        self._embedder = embedder or Embedder(self._cfg.embed_model)

    def retrieve(
        self,
        parsed_query: ParsedQuery,
        top_k: Optional[int] = None,
        qdrant_filter=None,
    ) -> list[ScoredCandidate]:
        """
        Run retrieval for *parsed_query* and return up to *top_k* candidates.

        Args:
            parsed_query:   Output from QueryParser.
            top_k:          Override default_top_k from config.
            qdrant_filter:  Pre-built Qdrant Filter (avoids rebuilding).
        """
        k = top_k or self._cfg.default_top_k
        query_text = parsed_query.semantic_query or ""

        if not query_text:
            logger.debug("Empty semantic query — running filter-only scroll.")
            return self._filter_only(qdrant_filter, k)

        # Embed query
        dense_vec = self._embedder.embed_dense(query_text)
        sp_idx, sp_val = self._embedder.embed_sparse(query_text)

        # Try hybrid (Prefetch + RRF)
        try:
            candidates = self._hybrid_query(
                dense_vec, sp_idx, sp_val, qdrant_filter, k
            )
            logger.debug("Hybrid retrieval returned %d candidates.", len(candidates))
            return candidates
        except Exception as exc:
            logger.warning(
                "Hybrid retrieval failed (%s) — falling back to dense-only.", exc
            )
            return self._dense_only(dense_vec, qdrant_filter, k)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _hybrid_query(
        self,
        dense_vec: list[float],
        sp_idx: list[int],
        sp_val: list[float],
        filt,
        top_k: int,
    ) -> list[ScoredCandidate]:
        from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

        client = self._store._get_client()
        collection = self._cfg.qdrant_collection
        fetch_limit = min(top_k * 4, 200)

        fusion = (
            Fusion.RRF
            if self._cfg.fusion_method.lower() != "dbsf"
            else Fusion.DBSF
        )

        prefetch = [
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=fetch_limit,
                filter=filt,
            ),
            Prefetch(
                query=SparseVector(indices=sp_idx, values=sp_val),
                using="sparse",
                limit=fetch_limit,
                filter=filt,
            ),
        ]

        results = client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=fusion),
            limit=top_k,
            with_payload=True,
        )

        return [
            ScoredCandidate(
                item=_payload_to_item(r.payload or {}),
                rrf_score=r.score,
            )
            for r in results.points
            if r.payload
        ]

    def _dense_only(
        self,
        dense_vec: list[float],
        filt,
        top_k: int,
    ) -> list[ScoredCandidate]:
        from qdrant_client.models import NamedVector

        client = self._store._get_client()
        results = client.search(
            collection_name=self._cfg.qdrant_collection,
            query_vector=NamedVector(name="dense", vector=dense_vec),
            query_filter=filt,
            limit=top_k,
            with_payload=True,
        )
        return [
            ScoredCandidate(
                item=_payload_to_item(r.payload or {}),
                rrf_score=r.score,
                dense_score=r.score,
            )
            for r in results
            if r.payload
        ]

    def _filter_only(self, filt, top_k: int) -> list[ScoredCandidate]:
        """Scroll collection with payload filter when there is no semantic query."""
        client = self._store._get_client()
        points, _ = client.scroll(
            collection_name=self._cfg.qdrant_collection,
            scroll_filter=filt,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            ScoredCandidate(item=_payload_to_item(p.payload or {}), rrf_score=1.0)
            for p in points
            if p.payload
        ]
