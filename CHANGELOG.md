# Changelog

All notable changes to the Recommendation Engine are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [0.3.0] — 2026-05-31

### Added — Phase 9: Quality, Tooling & Watch-History Feedback

#### Test infrastructure

- **`tests/conftest.py`** — Shared pytest fixtures powering the new test suite.
  Key pieces:
  - `FakeQdrantClient`: stateful in-memory Qdrant substitute with `create_collection`,
    `upsert`, `scroll`, `search`, `query_points`, `delete`, and call-log inspection via
    `client.calls`.
  - `FakeQdrantModule` / `FakeModelsModule`: full stub tree for `qdrant_client` and
    `qdrant_client.models` injected into `sys.modules` so all lazy imports resolve.
  - `FakeAnthropicModule`: minimal sync + async-capable fake with configurable response
    JSON; supports `await async_client.messages.create(...)`.
  - `MockEmbedder`: hash-based deterministic vectors (no BGE-M3 load); covers `embed_dense`,
    `embed_sparse`, and `embed_batch`.
  - Fixtures: `qdrant_mock`, `anthropic_mock`, `mock_embedder`, `sample_items` (3 validated
    MediaItems), `scored_candidates`, `cfg` (no-key Settings).

- **`tests/test_store.py`** — 13 tests. Verifies `create_collection` idempotency, payload
  index creation for all 7 filterable fields, `upsert` call delegation, batch-size splitting,
  `collection_info` count, `delete` removal.

- **`tests/test_retriever.py`** — 10 tests. Covers `_payload_to_item` edge cases (zero-valued
  optional fields, entity list, missing fields), hybrid query path, filter-only scroll for
  empty semantic query, top-K limiting, empty collection.

- **`tests/test_explainer.py`** — 8 tests. No-API-key fallback (all template reasons),
  `_fallback_reason` structure, consume-verb correctness, `matched_tags` Pydantic guard,
  mocked async path result count, rank preservation after explain_batch.

- **`tests/test_pipeline.py`** — 8 tests. Wires a complete pipeline from mocked parts;
  verifies sorted output, rank-1 assignment, top-K cap, `explain=False` produces empty
  reasons/tags, empty collection → empty list, component scores present.

- **`tests/test_output.py`** — 11 tests. `to_json` valid JSON, required keys, nested
  `component_scores`, empty input, indent parameter; `print_table` no-crash, missing
  links, score bar bounds, type badge for known + unknown types.

- **Total: 105 tests, 0 failures, 0 warnings.**

#### Prompt caching

- **`src/query_parser.py`** — `_call_claude` now passes a structured `system` list with
  `"cache_control": {"type": "ephemeral"}` on the `_SYSTEM_PROMPT` block. The 3 KB system
  prompt is transmitted once and cached for 5 minutes, cutting latency and token cost on
  repeated query-parse calls.

- **`src/explainer.py`** — Added module-level `_CACHED_SYSTEM` list (same `cache_control`
  pattern). Reused for every `_explain_one` call within a batch so the prompt is cached
  server-side across all parallel async requests.

#### Watch-history implicit feedback

- **`src/schema.py`** — `HistoryProfile` Pydantic model:
  - `preferred_tags: frozenset[str]`, `preferred_genres: frozenset[str]`, `item_count: int`
  - `from_payloads(payloads)` — builds profile from a list of Qdrant point payloads
  - `overlap_fraction(item)` — fraction of item's tags + genres that appear in the profile

- **`src/config.py`** — Two new settings:
  - `history_min_rating: float` (default 7.0) — minimum rating to include in profile
  - `history_boost_weight: float` (default 0.15) — boost strength (0 = off, 1 = max)

- **`src/scorer.py`** — `_history_boost(item, profile)`:
  - `boost = 1 + weight × overlap_fraction(item, profile)`
  - Neutral (1.0) when no profile or zero-item count
  - Wired into `_rank_one` as a fourth multiplicative factor
  - `score()` now accepts `history_profile: Optional[HistoryProfile]`
  - 5 new tests in `test_scorer.py` (empty profile, full overlap, partial overlap,
    integration with `score()`)

- **`src/pipeline.py`** — `_build_history_profile()` method:
  - Scrolls Qdrant for items with `watch_status in ["watched", "reading"]` AND
    `rating >= history_min_rating` (max 500 items)
  - Returns `None` on any error so the pipeline continues without the boost
  - `use_history: bool = True` constructor flag (disable with `--no-history`)
  - `_store` renamed from local `store` so `_build_history_profile` can access the client

#### CLI additions (`src/cli.py`)

- **`recommend sync --input FILE [--batch-size N]`** — Incremental upsert: validates input,
  embeds changed/new items, upserts to Qdrant (UUID-idempotent). Existing items are updated;
  items not in the file are untouched. Identical flow to `ingest` without `--reset`.

- **`recommend delete UUID [--yes]`** — Remove a single item from the collection. Prompts
  for confirmation unless `--yes` is passed.

- **`recommend query --no-history`** — Disable the watch-history boost for a single query
  (useful for comparing results with and without personal feedback signal).

#### Evaluation tooling

- **`tests/golden_queries.json`** — All 10 entries now carry pre-parsed `semantic_query`
  and `parsed_filters` fields (matching the QueryParser output schema), enabling
  deterministic evaluation without a Claude API key.

- **`scripts/evaluate.py`** (rewrite) — Uses pre-parsed filters directly via
  `_build_qdrant_filter()`; no Claude dependency. Added `--csv PATH` flag to export
  per-query metrics (NDCG@K, Precision@5, penalty count, result IDs) for post-processing.

- **`scripts/sweep.py`** (new) — Grid search over `lambda_recency ∈ [0.02, 0.05, 0.10]` ×
  `fusion_method ∈ [rrf, dbsf]` (6 combinations). For each combination, runs all golden
  queries through `HybridRetriever + Scorer` and computes mean NDCG@K. Prints a Rich
  comparison table with the best combination highlighted; optionally exports to CSV.

#### Documentation

- **`README.md`** (new) — Satisfies the Phase 0 requirement. Covers: how it works (pipeline
  diagram), 5-step quickstart (docker, install, .env, ingest, query), full command reference
  with flags table, data format with field descriptions, config table for all 14 env vars,
  development commands, test coverage table, and architecture decision notes.

- **`pyproject.toml`** — Removed `asyncio_mode = "auto"` (was causing a spurious pytest
  config warning because `pytest-asyncio` is an optional dev dependency).

- **`ROADMAP.md`** — Added Phase 9 with full task list and acceptance criteria; updated file
  structure section to reflect current state; moved "Watch history" from Open Questions to
  implemented (✅); added two new open questions (streaming explanations, collaborative
  filtering).

### Changed

- `src/scorer.py` docstring updated to include the fourth factor (`history_boost`).
- `src/pipeline.py` — `store` local renamed to `self._store`; `use_history` constructor
  parameter added; `_build_history_profile` defined as an instance method.
- `.env.example` — Added `HISTORY_MIN_RATING` and `HISTORY_BOOST_WEIGHT` entries.

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
