# Changelog

All notable changes to the Recommendation Engine are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [0.2.0] — 2026-05-31

### Added — Phase 2: Query Parser

- **`src/cache.py`** — Thread-safe in-memory LRU cache keyed on normalised
  (lowercase, collapsed-whitespace) query strings. Prevents redundant Claude
  API calls for repeated or near-identical queries.
- **`src/query_parser.py`** — `QueryParser` class wrapping the Claude API
  (`claude-sonnet-4-6` by default). Ships a 10-example few-shot system prompt
  covering: pure-semantic, pure-filter, mixed NL+structural, manga volumes,
  CS paper, classic movie, plan-to-watch list, rating threshold, reading status,
  and short-series-with-genre queries. Outputs `ParsedQuery` with
  `semantic_query`, `filters`, and `length_preference_episodes`.
  `_build_qdrant_filter()` converts `FilterClause` objects into Qdrant `Filter`
  using `must`/`must_not` lists with `MatchValue`, `MatchAny`, and `Range`
  conditions.
- **Three-layer fallback** in `QueryParser`: missing API key → semantic-only;
  malformed JSON from Claude → semantic-only; any API/network error →
  semantic-only. The pipeline never crashes due to a parsing failure.

### Added — Phase 3: Hybrid Retrieval

- **`src/retriever.py`** — `HybridRetriever` class. Issues a single Qdrant
  `query_points` call with two `Prefetch` legs (dense cosine + sparse dot-product)
  fused server-side with RRF or DBSF (configurable via `FUSION_METHOD`).
  Fetches `top_k × 4` candidates per leg to ensure the fusion has enough
  material. Falls back to `client.search()` with a `NamedVector` on any
  `Prefetch`/fusion error. Supports filter-only scroll when `semantic_query`
  is empty (handles plan-to-watch-list style queries with no semantic content).

### Added — Phase 4: Score Enhancement

- **`src/scorer.py`** — `Scorer` class computing a multiplicative
  **Recommendation Value**: `rrf_score × rating_boost × recency_decay × length_decay`.
  - `rating_boost = 1 + log(1 + rating/10)` — logarithmic; 10/10 ≈ 1.69×, no rating = 1.0×.
  - `recency_decay = exp(-λ × age_years)` — configurable `LAMBDA_RECENCY` (default 0.05).
  - `length_decay = exp(-0.5 × ((eps − pref) / scale)²)` — Gaussian; only applied when
    `ParsedQuery.length_preference_episodes` is set; 1.0 otherwise.
  - Results sorted descending by `recommendation_value`; 1-based `rank` assigned after sort.
  - `ComponentScores` embedded in each `RankedResult` for transparency.

### Added — Phase 5: Explainability (XAI)

- **`src/explainer.py`** — `Explainer` class with `explain_batch()`. Fires
  all explanations in parallel via `asyncio.gather` and `anthropic.AsyncAnthropic`,
  keeping total latency ≈ single API call. System prompt enforces: cite only
  provided metadata, no plot spoilers, modality-aware consume verb ("read" vs
  "watch"), 2–4 reasons, `matched_tags` must be a strict subset of actual tags.
  Falls back to a template-based `ExplainedResult` on any per-item error.
- **Anti-hallucination guard** in `ExplainedResult.validate_matched_tags()`
  (Pydantic v2 `@model_validator`): strips any tag in `matched_tags` not
  present in the item's actual `tags + genres` at model construction time.

### Added — Phase 6: CLI & Output Layer

- **`src/pipeline.py`** — `RecommendationPipeline` wiring all stages. A shared
  `Embedder` instance avoids loading BGE-M3 twice. Constructor flags: `top_k`,
  `use_reranker`, `explain`. Each stage logs timing and candidate counts.
- **`src/output.py`** — Two renderers: `print_table()` (Rich table with score
  bar, type badges, truncated links, reasons, matched tags) and `to_json()`
  (JSON-serialisable list of dicts including component scores).
- **Updated `src/cli.py`** — Added `-v/--verbose` flag for DEBUG logging.
  `query` command is now fully functional: `--top-k`, `--format table|json`,
  `--rerank`, `--no-explain`. `info` command prints full config including
  Claude model, fusion method, and λ recency.

### Added — Phase 7: Multi-Modal Expansion

- **`src/schema.py`** — `MediaItem` gains `abstract: Optional[str]` (primary
  dense source for papers) and `volumes: Optional[int]` (manga). `dense_text`
  property dispatches: `review_notes → abstract → synthesised fallback`.
  Added `consume_verb` property ("read"/"watch") and `is_video` property.
- **`data/sample_books.json`** — 5 book entries: Dune, The Left Hand of
  Darkness, Flowers for Algernon, Blindsight, The Three-Body Problem.
- Query parser few-shot examples cover manga volume filter and CS paper queries.
- Explainer system prompt now references `consume_verb` instruction.

### Added — Phase 8: Reranking & Evaluation

- **`src/reranker.py`** — `Reranker` class wrapping `FlagEmbedding.FlagReranker`
  (`bge-reranker-v2-m3`). Thread-safe singleton loader. `normalize=True` for
  sigmoid-mapped scores. Replaces `rrf_score` in each `ScoredCandidate` with
  the cross-encoder score. Falls back to original ordering on any model error.
- **`tests/golden_queries.json`** — 10 labelled query/result pairs with
  `expected_top_ids` and `expected_absent_ids` for NDCG/Precision evaluation.
- **`scripts/evaluate.py`** — Click CLI computing NDCG@K and Precision@5
  against the golden set. Penalty counter tracks `expected_absent_ids` that
  appeared in results. Rich table output with per-query and mean metrics.
  `--rerank` flag enables comparison with/without cross-encoder.

### Changed

- **`src/config.py`** — Replaced deprecated `Field(env=...)` with
  `Field(validation_alias=...)` (pydantic-settings ≥ 2.0 API). Added new
  settings: `length_origin`, `length_scale`, `fusion_method`, `rerank_model`,
  `rerank_top_n`, `claude_model`.
- **`.env.example`** — Documents all new config variables.
- **`ROADMAP.md`** — All Phase 0–8 task lists updated with ✅ completion
  markers; file structure section updated to reflect current state.

### Tests

- **`tests/test_query_parser.py`** — 29 tests (10 query types, 3 fallback
  scenarios, 4 cache tests, 5 filter-building tests). All dependencies
  (`anthropic`, `qdrant_client`) are mocked via `sys.modules` injection so
  tests run without the packages installed.
- **`tests/test_scorer.py`** — 12 tests covering all three decay/boost
  functions individually plus integration scenarios.
- **Total: 53 tests, all passing.**

---

## [0.1.0] — 2026-05-30

### Added — Phase 0: Project Foundation

- **`pyproject.toml`** — Project metadata and dependencies: `qdrant-client`,
  `FlagEmbedding`, `anthropic`, `langchain-core`, `click`, `pydantic`,
  `pydantic-settings`, `rich`, `python-dotenv`.
- **`.env.example`** — Template with all required and optional environment
  variables documented.
- **`docker-compose.yml`** — Local Qdrant service on port 6333 with persistent
  volume.
- **`src/schema.py`** — Pydantic v2 `MediaItem` with strict validators (rating
  0–10, year 1800–2200, non-negative episodes), CSV→list coercion for
  `genres`/`tags`/`associated_entities`, `dense_text` and `sparse_text`
  computed properties. Also defines `EmbeddedItem`, `FilterClause`,
  `ParsedQuery`, `ScoredCandidate`, `ComponentScores`, `RankedResult`,
  `ExplainedResult` with `matched_tags` anti-hallucination validator.
- **`src/config.py`** — `Settings` via pydantic-settings; local vs remote
  Qdrant toggle; all model and decay parameters.
- **`data/sample.json`** — 10 hand-crafted items (Evangelion, Perfect Blue,
  Steins;Gate, Violet Evergarden, Cowboy Bebop, Vinland Saga, Serial
  Experiments Lain, FMA:Brotherhood, Paranoia Agent, Attack on Titan).

### Added — Phase 1: Ingestion Pipeline

- **`src/embedder.py`** — `Embedder` wrapping `BGEM3FlagModel`. Thread-safe
  process-level singleton. `embed_dense()`, `embed_sparse()`, `embed_batch()`
  with optional progress callback.
- **`src/store.py`** — `QdrantStore` with idempotent `create_collection()`
  (creates multi-vector collection + 7 payload indexes), `upsert()` (batch),
  `delete()`, `collection_info()`.
- **`src/ingest.py`** — `recommend ingest` CLI with Pydantic validation, Rich
  progress bar, `--reset` flag to drop and recreate the collection.
- **`src/export.py`** — `recommend export` CLI using Qdrant scroll API.
- **`src/cli.py`** — Click group entry point; `ingest` and `export` wired in;
  `info` command showing collection stats.

### Tests

- **`tests/test_schema.py`** — 12 tests validating all 10 sample items, CSV
  coercion, rating bounds, `dense_text` dispatch, `sparse_text` content,
  `matched_tags` stripping, `ParsedQuery` defaults.
