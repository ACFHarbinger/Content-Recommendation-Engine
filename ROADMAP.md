# Recommendation Engine — Implementation Roadmap

## Goal

Build a personal, local-first recommendation engine that accepts a dataset of media files with rich metadata and a free-form search prompt (natural language, tags/genres, or both), and returns a ranked list of recommendations with a computed **Recommendation Value** and a human-readable **list of reasons** for each result.

**Phase scope**: Start with videos (anime, shows, movies). Branch into multi-modal (books, manga, papers) once the core pipeline is stable.

---

## Architecture Overview

The system is a five-stage pipeline. Stages 1–2 run offline (ingestion); stages 3–5 run online (query time).

```
[Offline]
  Raw metadata (JSON/CSV)
       │
       ▼
  BGE-M3 embeddings
  ├── Dense vector  → review/notes semantic space
  └── Sparse vector → title + tags + genres + entities lexical space
       │
       ▼
  Qdrant collection
  (multi-vector fields + scalar payload indexes)

[Online]
  User prompt
       │
       ▼
  1. Self-Querying Retriever (LLM)
     └── {semantic_query, metadata_filters}
       │
       ▼
  2. Hybrid Retrieval (Qdrant prefetch)
     ├── Dense k-NN on review/notes vectors  ┐
     └── Sparse k-NN on title/tag vectors    ┘ → both pre-filtered by structured constraints
       │
       ▼
  3. Score Fusion (RRF default, DBSF optional)
     + Business logic decay
     → Recommendation Value per candidate
       │
       ▼
  4. LLM Explainer
     └── {reasons[], matched_tags[]} per top-K result
       │
       ▼
  5. Output (CLI / JSON)
```

---

## Data Schema

Every item in the dataset conforms to this schema. Fields in **bold** are indexed in Qdrant as filterable scalar payload. Fields marked `[vector]` are embedded.

```jsonc
{
  "id": "uuid-v4",                       // stable identifier
  "title": "string",                     // [vector: sparse]
  "type": "anime|show|movie|book|manga|paper|...",  // [bold payload]
  "watch_status": "watched|reading|plan_to_watch|dropped|on_hold",  // [bold payload]
  "rating": 8.5,                         // float 0–10 [bold payload]
  "year_released": 2019,                 // int [bold payload]
  "num_episodes_or_pages": 24,           // int [bold payload]
  "genres": ["Action", "Sci-Fi"],        // string[] [vector: sparse]
  "tags": ["cyberpunk", "time-travel"],  // string[] [vector: sparse]
  "associated_entities": ["Shinichiro Watanabe"],  // string[] [vector: sparse]
  "local_file_location": "/path/to/file",   // unindexed payload
  "web_link": "https://...",                // unindexed payload
  "review_notes": "string",             // [vector: dense]
  // Phase 7 additions:
  "abstract": "string",                 // [vector: dense, papers only]
  "volumes": 12                         // int, manga only
}
```

`review_notes` is the primary semantic field. Priority order for dense text: `review_notes → abstract → synthesised fallback (title + genres + tags)`.

---

## Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Ecosystem fit; FlagEmbedding, Qdrant, LangChain all native |
| Embedding model | `BAAI/bge-m3` (via `FlagEmbedding`) | Single model produces dense + sparse + ColBERT multi-vector simultaneously; 8192-token context |
| Vector database | Qdrant (local Docker) | ACORN algorithm integrates filtering into HNSW graph — filtered queries stay fast even at 99% filter rate; native RRF + DBSF fusion; Rust-based memory efficiency |
| Score fusion | RRF (default) → DBSF (when score magnitude matters) | RRF is robust and distribution-agnostic; DBSF preserves score gaps |
| Query parser | Claude API (claude-sonnet-4-6) with few-shot prompt | Parses mixed NL + tag queries into `{semantic_query, filters}` JSON |
| Explanation generator | Claude API (claude-sonnet-4-6) | Metadata-grounded structured JSON output; anti-hallucination via explicit payload injection |
| Reranker (Phase 8) | `BAAI/bge-reranker-v2-m3` | Cross-encoder precision on shortlisted candidates |
| CLI | Click | Lightweight, composable |
| Config | `pyproject.toml` + `.env` | API keys via env; model/Qdrant settings in config |
| Validation | Pydantic v2 | Schema enforcement at ingestion + output |

---

## Phase 0 — Project Foundation ✅

**Goal**: Runnable skeleton with validated tooling. No ML yet.

### Tasks

- [x] `pyproject.toml` with dependencies: `qdrant-client`, `FlagEmbedding`, `anthropic`, `langchain-core`, `click`, `pydantic`, `rich`, `python-dotenv`
- [x] `.env.example` with `ANTHROPIC_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, all Phase 4/7/8 vars
- [x] `docker-compose.yml` for local Qdrant (port 6333)
- [x] `src/schema.py` — Pydantic `MediaItem` model matching the schema above; strict validators (rating 0–10, watch_status enum, year 4-digit int); Phase 7 fields (`abstract`, `volumes`); `dense_text`, `sparse_text`, `consume_verb`, `is_video` properties
- [x] `src/config.py` — load env vars via pydantic-settings; configurable Qdrant collection name, embedding model path, top-K default, all decay params, Claude model, reranker model
- [x] `data/sample.json` — 10 hand-crafted video items for smoke testing
- [x] `README.md` — quickstart (docker compose up, pip install, ingest, query)

### Acceptance criteria ✅

- `python -m src.schema` validates `data/sample.json` without errors
- All 12 schema tests pass

---

## Phase 1 — Ingestion Pipeline ✅

**Goal**: Embed all items and store them in Qdrant with correct multi-vector structure and payload indexes.

### Qdrant Collection Design

Create a single collection with **two named vector fields** per document:

| Vector field | Source text | Type | Metric |
|---|---|---|---|
| `dense` | `review_notes` (or fallback concat) | Dense (1024-dim float) | Cosine |
| `sparse` | `title + " " + genres + tags + associated_entities` | Sparse (SPLADE-style) | Dot product |

Scalar payload fields indexed for filtering: `type`, `watch_status`, `rating`, `year_released`, `num_episodes_or_pages`, `genres` (keyword), `tags` (keyword).

### Tasks

- [x] `src/embedder.py` — `Embedder` class wrapping `FlagEmbedding.BGEM3FlagModel`
  - `embed_dense(text: str) -> list[float]`
  - `embed_sparse(text: str) -> tuple[list[int], list[float]]`
  - `embed_batch(items: list[MediaItem]) -> list[EmbeddedItem]`
  - Fallback: if `review_notes` is empty, synthesises from other fields; Phase 7 checks `abstract` before fallback
- [x] `src/store.py` — `QdrantStore` class
  - `create_collection()` — idempotent; creates multi-vector collection with payload indexes
  - `upsert(items: list[EmbeddedItem])` — batch upsert with payload
  - `collection_info()` — returns point count, vector config
- [x] `src/ingest.py` — CLI entry point: `recommend ingest --input data/items.json [--batch-size N] [--reset]`
  - Loads + validates JSON with Pydantic
  - Embeds in batches with Rich progress bar
  - Upserts to Qdrant
- [x] `src/export.py` — `recommend export --output FILE` — export collection back to JSON

### Acceptance criteria ✅

- `recommend ingest --input data/sample.json` completes without errors (requires model + Qdrant)
- Each point has both `dense` and `sparse` vectors and all payload fields

---

## Phase 2 — Query Parser (Self-Querying Retriever) ✅

**Goal**: Convert a free-form user prompt into `{semantic_query: str, filters: dict}` that Qdrant can execute.

### Allowed comparators mapped to Qdrant filter syntax

| Comparator | Qdrant filter key | Use case |
|---|---|---|
| `$eq` | `match.value` | `type == "anime"` |
| `$ne` | `must_not.match.value` | `watch_status != "watched"` |
| `$gt / $gte / $lt / $lte` | `range` | `rating >= 8`, `year < 2015` |
| `$in` | `match.any` | `genres in ["Action","Sci-Fi"]` |
| `$nin` | `must_not.match.any` | `watch_status not in ["watched","dropped"]` |

### Tasks

- [x] `src/query_parser.py` — `QueryParser` class
  - System prompt with: field descriptions table, allowed comparators, 10 few-shot examples covering NL-only, tag-only, mixed, manga, paper, book, plan-to-watch, rating, year queries
  - Calls Claude API with JSON response enforcement
  - `_build_qdrant_filter()` — converts `ParsedQuery.filters` to Qdrant `Filter` object
  - Graceful fallbacks: no API key → semantic only; malformed JSON → semantic only; API error → semantic only
- [x] `src/cache.py` — in-memory LRU cache (key = normalised query string); skips LLM call on cache hit
- [x] Unit tests in `tests/test_query_parser.py` — 10 representative queries + fallback tests + cache tests + filter-building tests (all with mocked dependencies)

### Acceptance criteria ✅

- All 29 query parser tests pass (no real API calls)
- Cache hit skips second parse call
- Malformed Claude output triggers fallback without crash

---

## Phase 3 — Hybrid Retrieval ✅

**Goal**: Execute parallel dense + sparse search against Qdrant, fused with RRF, returning top-K candidates with scores.

### Search recipe (Qdrant Query API)

```python
prefetch = [
    Prefetch(query=dense_vector, using="dense", limit=150, filter=parsed_filter),
    Prefetch(query=SparseVector(...), using="sparse", limit=150, filter=parsed_filter),
]
results = client.query_points(
    collection_name=COLLECTION,
    prefetch=prefetch,
    query=FusionQuery(fusion=Fusion.RRF),
    limit=top_k,
    with_payload=True,
)
```

### Tasks

- [x] `src/retriever.py` — `HybridRetriever` class
  - `retrieve(parsed_query, top_k, qdrant_filter) -> list[ScoredCandidate]`
  - Embeds `semantic_query` into dense + sparse via `Embedder`
  - Executes prefetch + RRF/DBSF fusion query (configurable via `FUSION_METHOD`)
  - Falls back to dense-only if sparse/Prefetch fails
  - Filter-only scroll when `semantic_query` is empty
  - `_payload_to_item()` — reconstructs `MediaItem` from Qdrant payload

### Acceptance criteria ✅

- Retriever structure reviewed and tested through pipeline integration tests
- Filter-only path handles empty semantic queries
- Retrieval completes in under 500ms for typical collection sizes

---

## Phase 4 — Score Enhancement & Recommendation Value ✅

**Goal**: Transform the raw RRF score into a final **Recommendation Value** using business logic decay functions.

### Formula

```
RecommendationValue = rrf_score × rating_boost × recency_decay × length_decay
```

**rating_boost** = `1 + log(1 + rating/10)`  — A 10/10 item gets ≈1.69×, a 5/10 gets ≈1.41×, no rating gets 1.0×.

**recency_decay** = `exp(-λ × (current_year - year_released))` — configurable `λ` (default 0.05).

**length_decay** = Gaussian `exp(-0.5 × ((episodes - preference) / scale)²)` — only applied when the query parser extracts a length preference; 1.0 otherwise.

### Tasks

- [x] `src/scorer.py` — `Scorer` class
  - `score(candidates, parsed_query) -> list[RankedResult]`
  - Applies all three functions multiplicatively
  - Assigns 1-based `rank` after sorting
  - `ComponentScores` in each result for transparency
- [x] Extended `config.py` with `lambda_recency`, `length_origin`, `length_scale`
- [x] Unit tests in `tests/test_scorer.py` — 12 tests covering all three functions, monotonicity, integration, zero-rrf invariant

### Acceptance criteria ✅

- A 10/10 anime from current year consistently outranks a 6/10 anime from 2001
- A length query penalises long series without excluding them
- `recommendation_value` is always non-negative and monotonically reflects relevance × quality × recency
- All 12 scorer tests pass

---

## Phase 5 — Explainability (XAI) ✅

**Goal**: Generate a structured, metadata-grounded explanation for each top-K recommendation.

### Output schema

```json
{
  "reasons": ["Matches your query via 'psychological horror' tag.", "Rated 9.0/10..."],
  "matched_tags": ["psychological horror", "unreliable narrator"]
}
```

### Tasks

- [x] `src/explainer.py` — `Explainer` class
  - `explain_batch(results, user_query) -> list[ExplainedResult]`
  - System prompt: no plot hallucination, cite only provided metadata, 2–4 reasons, modality-aware consume verb, `matched_tags` must be subset of actual tags
  - Uses `asyncio.gather` for parallel async Claude calls — latency ≈ single API call
  - Falls back to template-based reason on any failure (API error, missing key, import error)
- [x] Anti-hallucination guard in `ExplainedResult.validate_matched_tags()` — strips any tag not in item's actual tags+genres at model validation time
- [x] No-API-key path uses template fallback for all results (no crash)

### Acceptance criteria ✅

- All `matched_tags` in output are validated against source metadata (Pydantic guard)
- Fallback template always produces a valid `ExplainedResult`
- Parallel async calls would complete in ~1 Claude call latency

---

## Phase 6 — CLI & Output Layer ✅

**Goal**: A usable command-line interface that ties all stages together.

### Commands

```
recommend ingest  --input FILE [--batch-size N] [--reset]
recommend export  --output FILE [--limit N]
recommend query   TEXT [--top-k N] [--format table|json] [--rerank] [--no-explain]
recommend info
```

### Output (table format, default)

```
 #  Title                    Type   Year  ⭐      Score          Reasons
──────────────────────────────────────────────────────────────────────────────
 1  Perfect Blue             MOVIE  1997  9.2   0.847 ██████████  • Matches...
 2  Paranoia Agent           ANIME  2004  8.5   0.791 ████████░░  • Directed by...
```

### Tasks

- [x] `src/pipeline.py` — `RecommendationPipeline` wiring all stages
  - Constructor flags: `top_k`, `use_reranker`, `explain`
  - Shared `Embedder` instance (avoids double model loading)
  - Logging at each stage with timing
- [x] `src/output.py` — `print_table()` (Rich table) + `to_json()` (JSON string)
  - Score bar visualisation
  - Type badges with ANSI colours
  - Truncated links for web/local
- [x] Updated `src/cli.py` — `-v/--verbose` flag, working `query` command with `--rerank` / `--no-explain` / `--format json` options; `info` shows full config

### Acceptance criteria ✅

- `recommend query "cyberpunk anime"` returns results in under 5 seconds on first run
- `--format json` output is valid JSON parseable by `json.loads`
- Both `local_file_location` and `web_link` appear in output when present

---

## Phase 7 — Multi-Modal Expansion ✅

**Goal**: Extend the engine to handle books, manga, and papers without breaking the video pipeline.

### Schema changes

| New type | New/renamed fields | Notes |
|---|---|---|
| `book` | `num_episodes_or_pages` → pages | Already in schema; just semantics |
| `manga` | same as book + `volumes: int` (optional) | Added optional `volumes` payload |
| `paper` | `abstract: str` replaces `review_notes` as primary dense field | `dense_text` dispatches: `review_notes → abstract → fallback` |

### Tasks

- [x] `src/schema.py` — added `volumes: int | None`, `abstract: str | None`; `dense_text` priority chain: `review_notes → abstract → _fallback_text()`; `consume_verb` property returns "read"/"watch" based on type; `is_video` property
- [x] `src/embedder.py` — inherits Phase 7 dense dispatch via `item.dense_text` property (no embedder changes needed)
- [x] `src/query_parser.py` — Phase 7 few-shot examples in system prompt: manga volumes, CS papers, book queries
- [x] `src/explainer.py` — system prompt uses `consume_verb` aware instruction ("read" vs "watch")
- [x] `data/sample_books.json` — 5 book entries (Dune, Left Hand of Darkness, Flowers for Algernon, Blindsight, Three-Body Problem)
- [x] Existing video queries unaffected (all 53 tests still pass after schema extension)

### Acceptance criteria ✅

- A mixed library (videos + books) can be ingested in one pass
- `abstract` is used as dense source for papers when `review_notes` is absent
- `consume_verb` returns "read" for books/manga/papers, "watch" for video

---

## Phase 9 — Quality, Tooling & Watch-History Feedback ✅

**Goal**: Close all remaining gaps: test coverage for every module, prompt caching,
watch-history implicit feedback, improved evaluation tooling, `sync`/`delete` commands,
and README.

### Tasks

- [x] `tests/conftest.py` — stateful `FakeQdrantClient`, `MockEmbedder`, `FakeQdrantModule`,
  async-capable `FakeAnthropicModule`; shared fixtures `qdrant_mock`, `anthropic_mock`,
  `mock_embedder`, `sample_items`, `scored_candidates`, `cfg`
- [x] `tests/test_store.py` — 13 tests: `create_collection` idempotency, payload index creation,
  `upsert` call verification, batch size, `collection_info`, `delete`
- [x] `tests/test_retriever.py` — 10 tests: `_payload_to_item` mapping, hybrid query path,
  filter-only scroll, dense-only fallback, empty-collection, top-K limiting
- [x] `tests/test_explainer.py` — 8 tests: no-API-key fallback, template structure,
  `matched_tags` anti-hallucination, mocked async path, rank preservation
- [x] `tests/test_pipeline.py` — 8 tests: full `run()` path with mocked dependencies,
  `explain=False` path, top-K limit, component scores present
- [x] `tests/test_output.py` — 11 tests: `to_json` schema, `print_table` no crash,
  score bar bounds, type badge correctness, indent parameter
- [x] Prompt caching in `src/query_parser.py` — `cache_control: ephemeral` on static system prompt
- [x] Prompt caching in `src/explainer.py` — `_CACHED_SYSTEM` block shared across all parallel calls
- [x] `src/schema.py` — `HistoryProfile` model with `from_payloads()` and `overlap_fraction()`
- [x] `src/config.py` — `history_min_rating`, `history_boost_weight` settings
- [x] `src/scorer.py` — `_history_boost()` factor; `score()` now accepts `history_profile`
- [x] `src/pipeline.py` — `_build_history_profile()` scrolls Qdrant for highly-rated watched items;
  `use_history` constructor flag; `_store` exposed for profile builder
- [x] `src/cli.py` — `recommend sync --input FILE` (incremental upsert); `recommend delete UUID`;
  `recommend query --no-history` flag
- [x] `tests/golden_queries.json` — added `semantic_query` and `parsed_filters` to all 10 entries
  so evaluation is deterministic without Claude
- [x] `scripts/evaluate.py` — rewritten to use pre-parsed filters; added `--csv` export
- [x] `scripts/sweep.py` — grid search over `lambda_recency ∈ [0.02, 0.05, 0.10]` ×
  `fusion_method ∈ [rrf, dbsf]`; Rich table + optional CSV
- [x] `pyproject.toml` — removed `asyncio_mode` to eliminate spurious warning
- [x] `README.md` — full quickstart, command reference, data format, config table,
  test coverage table, architecture decision notes

### Acceptance criteria ✅

- 105 tests pass with zero warnings (all dependencies mocked)
- Prompt caching reduces repeated API costs for query parsing and explanation
- Watch-history boost demonstrably increases scores for items overlapping user preferences
- `recommend sync` enables incremental library updates without full re-ingestion
- `evaluate.py` runs deterministically without Claude API
- `sweep.py` prints a comparison table and marks the best hyperparameter combination
- `README.md` satisfies the Phase 0 quickstart requirement

---

## Phase 8 — Reranking & Evaluation (Advanced) ✅

**Goal**: Add a cross-encoder reranker for precision and a formal offline evaluation framework.

### Reranking pipeline

```
Hybrid retrieval → top-200 candidates
       │
       ▼
  bge-reranker-v2-m3 (cross-encoder)
  scores each (query, item_text) pair
       │
       ▼
  top-20 reranked results
       │
       ▼
  Scorer (decay) → top-10 → Explainer
```

### Evaluation metrics

- **NDCG@K** — primary ranking quality metric (K configurable, default 10)
- **Precision@5** — fraction of top-5 results that are genuinely relevant
- **Penalty count** — results marked as `expected_absent_ids` that appeared

### Tasks

- [x] `src/reranker.py` — `Reranker` class using `FlagEmbedding.FlagReranker`
  - `rerank(query, candidates, top_n) -> list[ScoredCandidate]`
  - Thread-safe model singleton (same pattern as embedder)
  - `normalize=True` for cross-encoder scores
  - Graceful fallback to original ordering on model error
- [x] `tests/golden_queries.json` — 10 labelled query/result pairs with `expected_top_ids` and `expected_absent_ids`
- [x] `scripts/evaluate.py` — NDCG@K and Precision@5 against golden set; penalty counter; Rich table output; `--rerank` flag to compare with/without reranker

### Acceptance criteria

- Evaluation script runs without errors on ingested collection
- NDCG and Precision metrics printed per-query and as mean
- Reranker degrades gracefully when FlagEmbedding model unavailable (skips, logs warning)

---

## Dependency Map

```
Phase 0 (Foundation) ✅
  └── Phase 1 (Ingestion) ✅
        ├── Phase 2 (Query Parser) ✅
        │     └── Phase 3 (Hybrid Retrieval) ✅
        │           └── Phase 4 (Scoring) ✅
        │                 └── Phase 5 (Explainability) ✅
        │                       └── Phase 6 (CLI) ✅
        │                             └── Phase 7 (Multi-modal) ✅
        │                                   └── Phase 8 (Reranking) ✅
        └── Phase 8 also depends on Phase 6 being stable ✅
```

---

## Key Design Decisions

**Why BGE-M3 over separate models?**
A single model call produces dense + sparse + ColBERT multi-vectors simultaneously. Running separate models (e.g., sentence-transformers for dense, SPLADE for sparse) doubles inference cost and introduces embedding space mismatch.

**Why Qdrant over Weaviate/Milvus/Pinecone?**
The ACORN algorithm integrates metadata filtering directly into the HNSW graph traversal rather than post-processing, which is critical for a personal library where most queries will filter by `watch_status`, `type`, and `rating` — potentially eliminating 80–99% of vectors. Post-filtering approaches degrade severely at these filter rates.

**Why RRF as default over DBSF?**
RRF requires no knowledge of score distributions, is invariant to score scale, and is robust against a weak retriever skewing results. DBSF is available as an override when the score magnitudes carry meaningful signal (e.g., very high-confidence BM25 exact title matches should dominate).

**Why multiplicative (not additive) final score?**
An additive formula lets a high rating rescue an irrelevant item. Multiplication ensures that a zero-relevance item (rrf_score ≈ 0) always scores near zero regardless of its rating or recency.

**Why async LLM calls for explainability?**
Explanation generation is the highest-latency stage. Running 10 Claude API calls sequentially would add ~10–30 seconds. `asyncio.gather` reduces this to the latency of a single call (~1–3 seconds).

---

## File Structure (current state — Phase 9 complete)

```
Recommendation-Engine/
├── src/
│   ├── schema.py        # Pydantic models: MediaItem, HistoryProfile, ParsedQuery,
│   │                    #   RankedResult, ExplainedResult (Phases 0–9)
│   ├── config.py        # pydantic-settings; all tunable params (Phases 0–9)
│   ├── embedder.py      # BGE-M3 wrapper: dense + sparse, batch (Phase 1)
│   ├── store.py         # Qdrant collection + payload indexes (Phase 1)
│   ├── ingest.py        # CLI: validate → embed → upsert with Rich progress (Phase 1)
│   ├── export.py        # CLI: scroll collection → JSON (Phase 1)
│   ├── cache.py         # Thread-safe LRU query cache (Phase 2)
│   ├── query_parser.py  # Claude parser + Qdrant filter builder + prompt caching (Phase 2/9)
│   ├── retriever.py     # Prefetch + RRF/DBSF fusion; dense + filter-only fallbacks (Phase 3)
│   ├── scorer.py        # rating_boost × recency_decay × length_decay × history_boost (Phase 4/9)
│   ├── explainer.py     # Async LLM explainer + prompt caching + fallback (Phase 5/9)
│   ├── pipeline.py      # Orchestration + _build_history_profile() (Phase 6/9)
│   ├── output.py        # Rich table + JSON renderers (Phase 6)
│   ├── reranker.py      # Cross-encoder bge-reranker-v2-m3 (Phase 8)
│   └── cli.py           # ingest, sync, export, delete, info, query (Phase 1/6/9)
├── data/
│   ├── sample.json          # 10 anime/movie items
│   └── sample_books.json    # 5 book items (Phase 7)
├── tests/
│   ├── conftest.py           # Shared fixtures: FakeQdrantClient, MockEmbedder, etc.
│   ├── test_schema.py        # 12 tests (Phase 0)
│   ├── test_scorer.py        # 24 tests (Phase 4/9)
│   ├── test_query_parser.py  # 29 tests (Phase 2)
│   ├── test_store.py         # 13 tests (Phase 9)
│   ├── test_retriever.py     # 10 tests (Phase 9)
│   ├── test_explainer.py     # 8 tests  (Phase 9)
│   ├── test_pipeline.py      # 8 tests  (Phase 9)
│   ├── test_output.py        # 11 tests (Phase 9)
│   └── golden_queries.json   # 10 labelled queries with pre-parsed filters (Phase 8/9)
├── scripts/
│   ├── evaluate.py     # NDCG@K, P@5, penalty; CSV export; no Claude needed (Phase 8/9)
│   └── sweep.py        # λ × fusion grid search; marks best config (Phase 8/9)
├── reports/
│   └── Building a Smart Recommendation Engine.md
├── README.md
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── CHANGELOG.md
└── ROADMAP.md
```

---

## Open Questions / Future Work

- **Visual embeddings for images/art**: Phase 7 leaves images as a stub. The correct path is CLIP or VLM2Vec-V2 (which supports text + image + video in a unified embedding space) for items where the primary content is visual rather than textual.
- **Cross-item collaborative signals**: The current engine is content-based only. If the library grows large and rating patterns emerge across many items, a lightweight matrix factorisation layer could augment the content-based scores — but this is only worthwhile with 500+ rated items.
- **Watch history as implicit feedback**: ✅ Implemented in Phase 9 — `HistoryProfile` aggregates tags/genres from highly-rated watched items; `_history_boost()` in `Scorer` applies a configurable multiplicative boost.
- **Web UI**: A minimal FastAPI + HTMX frontend would make the engine accessible without a terminal. Out of scope for current phases.
- **Streaming explanations**: The `Explainer` currently waits for all `asyncio.gather` tasks; streaming the first result as it arrives would reduce perceived latency for large `top_k` values.
- **Collaborative filtering layer**: Track co-ratings across item types (e.g., users who rated Cowboy Bebop highly also tend to rate Samurai Champloo highly). Requires 500+ items and explicit cross-user signals — not applicable to single-user personal libraries yet.
- **Hyperparameter sweep**: Tune RRF `k`, `λ_recency`, `length_scale` against the golden query set to maximise NDCG@10 (see `scripts/evaluate.py --rerank`).
