# Roadmap — Search & Retrieval

The online query pipeline. Implementation in [`src/search/`](../../../src/search/): `query_parser.py`,
`retriever.py`, `scorer.py`, `reranker.py`, `explainer.py`, `pipeline.py`.

## §1 — Query parser (self-querying retriever)

- [x] LLM parses a free-form prompt into `{semantic_query, SQL WHERE clause}` (Anthropic Claude).
- [ ] Deterministic fallback parser when no API key is set; cache parsed queries.

## §2 — Hybrid retrieval

- [x] Cosine on dense vectors + dot-product on sparse vectors, both filtered by the SQL WHERE.
- [ ] Document the candidate cap and its latency/quality trade-off.

## §3 — Fusion & scoring

- [x] RRF fusion (default) with optional DBSF; business-logic decay → Recommendation Value.
- [ ] Make fusion weights/decay config-driven (`src/core/config.py`), not hard-coded.

## §4 — Reranking & evaluation (advanced)

- [x] Reranker stage wired into the pipeline.
- [ ] Offline evaluation harness (representative query set) gating reranker changes.

## §5 — Explainability (XAI)

- [x] LLM explainer emits `{reasons[], matched_tags[]}` per top-K result.
- [ ] Guard against hallucinated reasons: cite only tags/fields present on the item.
