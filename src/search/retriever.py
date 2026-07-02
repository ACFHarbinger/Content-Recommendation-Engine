"""
Phase 3 — Hybrid Retrieval: dense cosine-similarity + sparse dot-product
fused with Reciprocal Rank Fusion (RRF), backed by SQLite.

All vector arithmetic runs in Python — suitable for personal library sizes
(up to ~10 000 items).  Dense and sparse scores are computed independently
then merged with RRF, falling back to dense-only when no sparse data is
present (e.g. freshly ingested with sparse encoding disabled).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from src.core.config import Settings, get_settings # pyrefly: ignore [missing-import]
from src.data.embedder import Embedder # pyrefly: ignore [missing-import]
from src.core.schema import MediaItem, ParsedQuery, ScoredCandidate # pyrefly: ignore [missing-import]
from src.data.store import SQLiteStore # pyrefly: ignore [missing-import]

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Vector math helpers
# ------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _sparse_dot(
    idx_q: list[int],
    val_q: list[float],
    idx_d: list[int],
    val_d: list[float],
) -> float:
    doc = dict(zip(idx_d, val_d))
    return sum(v * doc.get(i, 0.0) for i, v in zip(idx_q, val_q))


def _rrf(ranks: list[int], k: int = 60) -> float:
    return sum(1.0 / (k + r) for r in ranks)


# ------------------------------------------------------------------
# Row → MediaItem
# ------------------------------------------------------------------


def _payload_to_item(payload: dict) -> MediaItem:
    """Convert a SQLite row dict into a MediaItem."""
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
    Retrieves top-K candidates from SQLite using hybrid dense+sparse search.

    Strategy:
      1. Embed semantic_query → dense vector + sparse vector.
      2. Compute cosine similarity (dense leg) and dot-product (sparse leg)
         against all rows matching the SQL filter.
      3. Rank each leg independently, then fuse with RRF (default) or DBSF.
      4. Return ScoredCandidate list.

    Falls back to dense-only when all items have empty sparse vectors.
    """

    def __init__(
        self,
        store: Optional[SQLiteStore] = None,
        embedder: Optional[Embedder] = None,
        settings: Optional[Settings] = None,
    ):
        self._cfg = settings or get_settings()
        self._store = store or SQLiteStore(self._cfg)
        self._embedder = embedder or Embedder(self._cfg.embed_model)

    def retrieve(
        self,
        parsed_query: ParsedQuery,
        top_k: Optional[int] = None,
        sql_filter: Optional[tuple] = None,
    ) -> list[ScoredCandidate]:
        """
        Run retrieval for *parsed_query* and return up to *top_k* candidates.

        Args:
            parsed_query:  Output from QueryParser.
            top_k:         Override default_top_k from config.
            sql_filter:    Pre-built (where_clause, params) tuple from
                           QueryParser.parse_to_sql(); avoids rebuilding.
        """
        k = top_k or self._cfg.default_top_k
        query_text = parsed_query.semantic_query or ""
        where_clause, params = sql_filter if sql_filter else ("", [])

        if not query_text:
            logger.debug("Empty semantic query — running filter-only scroll.")
            return self._filter_only(where_clause, params, k)

        dense_vec = self._embedder.embed_dense(query_text)
        sp_idx, sp_val = self._embedder.embed_sparse(query_text)

        try:
            candidates = self._hybrid_query(
                dense_vec, sp_idx, sp_val, where_clause, params, k
            )
            logger.debug("Hybrid retrieval returned %d candidates.", len(candidates))
            return candidates
        except Exception as exc:
            logger.warning(
                "Hybrid retrieval failed (%s) — falling back to dense-only.", exc
            )
            return self._dense_only(dense_vec, where_clause, params, k)

    # ------------------------------------------------------------------
    # Private retrieval paths
    # ------------------------------------------------------------------

    def _hybrid_query(
        self,
        dense_vec: list[float],
        sp_idx: list[int],
        sp_val: list[float],
        where_clause: str,
        params: list,
        top_k: int,
    ) -> list[ScoredCandidate]:
        rows = self._store.fetch_all(where_clause, params)
        if not rows:
            return []

        dense_scores = [
            (i, _cosine(dense_vec, row["dense_vector"])) for i, row in enumerate(rows)
        ]
        sparse_scores = [
            (
                i,
                _sparse_dot(
                    sp_idx, sp_val, row["sparse_indices"], row["sparse_values"]
                ),
            )
            for i, row in enumerate(rows)
        ]

        # Sort each leg by score (descending) and assign 1-based ranks
        dense_ranked = sorted(dense_scores, key=lambda x: x[1], reverse=True)
        sparse_ranked = sorted(sparse_scores, key=lambda x: x[1], reverse=True)
        dense_rank = {idx: r + 1 for r, (idx, _) in enumerate(dense_ranked)}
        sparse_rank = {idx: r + 1 for r, (idx, _) in enumerate(sparse_ranked)}

        use_dbsf = self._cfg.fusion_method.lower() == "dbsf"

        if use_dbsf:
            max_d = max(s for _, s in dense_scores) if dense_scores else 1.0
            max_s = max(s for _, s in sparse_scores) if sparse_scores else 1.0
            d_map = dict(dense_scores)
            s_map = dict(sparse_scores)
            fused = [
                (i, d_map[i] / (max_d or 1.0) + s_map[i] / (max_s or 1.0))
                for i in range(len(rows))
            ]
        else:
            fused = [
                (i, _rrf([dense_rank[i], sparse_rank[i]])) for i in range(len(rows))
            ]

        fused.sort(key=lambda x: x[1], reverse=True)
        fused = fused[:top_k]

        d_map = dict(dense_scores)
        s_map = dict(sparse_scores)
        return [
            ScoredCandidate(
                item=_payload_to_item(rows[i]),
                rrf_score=score,
                dense_score=d_map[i],
                sparse_score=s_map[i],
            )
            for i, score in fused
        ]

    def _dense_only(
        self,
        dense_vec: list[float],
        where_clause: str,
        params: list,
        top_k: int,
    ) -> list[ScoredCandidate]:
        rows = self._store.fetch_all(where_clause, params)
        if not rows:
            return []

        scored = sorted(
            (
                (i, _cosine(dense_vec, row["dense_vector"]))
                for i, row in enumerate(rows)
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [
            ScoredCandidate(
                item=_payload_to_item(rows[i]),
                rrf_score=score,
                dense_score=score,
            )
            for i, score in scored
        ]

    def _filter_only(
        self,
        where_clause: str,
        params: list,
        top_k: int,
    ) -> list[ScoredCandidate]:
        """Return metadata rows matching the filter, scored 1.0 (no ranking)."""
        rows = self._store.fetch_filtered(where_clause, params, limit=top_k)
        return [
            ScoredCandidate(item=_payload_to_item(row), rrf_score=1.0) for row in rows
        ]
