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
  "review_notes": "string"              // [vector: dense]
}
```

`review_notes` is the primary semantic field. If absent, it is auto-generated at ingestion time from the other fields (title + genres + tags + entities concatenated) so every item has a meaningful dense vector.

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

## Phase 0 — Project Foundation

**Goal**: Runnable skeleton with validated tooling. No ML yet.

### Tasks

- [ ] `pyproject.toml` with dependencies: `qdrant-client`, `FlagEmbedding`, `anthropic`, `langchain-core`, `click`, `pydantic`, `rich`, `python-dotenv`
- [ ] `.env.example` with `ANTHROPIC_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` (optional for local)
- [ ] `docker-compose.yml` for local Qdrant (port 6333)
- [ ] `src/schema.py` — Pydantic `MediaItem` model matching the schema above; strict validators (rating 0–10, watch_status enum, year 4-digit int)
- [ ] `src/config.py` — load env vars, configurable Qdrant collection name, embedding model path, top-K default
- [ ] `data/sample.json` — 5–10 hand-crafted video items for smoke testing
- [ ] `README.md` — quickstart (docker compose up, pip install, ingest, query)

### Acceptance criteria

- `python -m src.schema` validates `data/sample.json` without errors
- Qdrant health endpoint responds at `localhost:6333`

---

## Phase 1 — Ingestion Pipeline

**Goal**: Embed all items and store them in Qdrant with correct multi-vector structure and payload indexes.

### Qdrant Collection Design

Create a single collection with **two named vector fields** per document:

| Vector field | Source text | Type | Metric |
|---|---|---|---|
| `dense` | `review_notes` (or fallback concat) | Dense (1024-dim float) | Cosine |
| `sparse` | `title + " " + genres + tags + associated_entities` | Sparse (SPLADE-style) | Dot product |

Scalar payload fields indexed for filtering: `type`, `watch_status`, `rating`, `year_released`, `num_episodes_or_pages`, `genres` (keyword), `tags` (keyword).

Non-indexed payload (returned but not filtered): `local_file_location`, `web_link`, `title`, `id`.

### Tasks

- [ ] `src/embedder.py` — `Embedder` class wrapping `FlagEmbedding.BGEM3FlagModel`
  - `embed_dense(text: str) -> list[float]`
  - `embed_sparse(text: str) -> dict[int, float]` (token_id → weight from SPLADE head)
  - `embed_batch(items: list[MediaItem]) -> list[EmbeddedItem]`
  - Fallback: if `review_notes` is empty, synthesize from other fields
- [ ] `src/store.py` — `QdrantStore` class
  - `create_collection()` — idempotent; creates multi-vector collection with correct configs
  - `upsert(items: list[EmbeddedItem])` — batch upsert with payload
  - `collection_info()` — returns point count, vector config
- [ ] `src/ingest.py` — CLI entry point: `python -m src.ingest --input data/items.json`
  - Loads + validates JSON
  - Calls embedder in batches (default batch=32 to fit VRAM)
  - Upserts to Qdrant
  - Prints progress with `rich` progress bar
- [ ] `src/export.py` — export current collection back to JSON (useful for backup/migration)

### Acceptance criteria

- `python -m src.ingest --input data/sample.json` completes without errors
- Qdrant UI (`localhost:6333/dashboard`) shows correct point count
- Each point has both `dense` and `sparse` vectors and all payload fields

---

## Phase 2 — Query Parser (Self-Querying Retriever)

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

- [ ] `src/query_parser.py` — `QueryParser` class
  - System prompt includes: field descriptions table, allowed comparators, 5 few-shot examples covering NL-only, tag-only, and mixed queries
  - Few-shot examples must cover edge cases:
    - Pure semantic: `"something like Evangelion but less depressing"` → no filters, full semantic query
    - Pure filter: `"all anime I haven't watched yet"` → `watch_status != "watched"`, `type == "anime"`, empty semantic query
    - Mixed: `"short highly-rated mecha from the 90s I haven't seen"` → semantic `"mecha giant robots"` + 4 filters
  - Calls Claude API with `response_format` enforcing JSON output: `{semantic_query: str, filters: list[FilterClause]}`
  - Converts output to Qdrant `Filter` object (using `qdrant_client.models`)
- [ ] `src/cache.py` — in-memory LRU cache (key = normalized query string); skips LLM call on cache hit
- [ ] Unit tests in `tests/test_query_parser.py` — assert filter extraction for 10 representative queries without hitting the API (mock Claude responses)

### Accepted filter output format (internal)

```python
@dataclass
class ParsedQuery:
    semantic_query: str        # passed to embedder
    qdrant_filter: Filter      # passed directly to Qdrant search
    raw_filters: list[dict]    # logged for debugging
```

### Acceptance criteria

- Parser correctly extracts filters for all 10 test cases
- Cache hit avoids second API call for identical query
- Malformed Claude output triggers a fallback to pure semantic search (no crash)

---

## Phase 3 — Hybrid Retrieval

**Goal**: Execute parallel dense + sparse search against Qdrant, fused with RRF, returning top-K candidates with scores.

### Search recipe (Qdrant Query API)

```python
# Prefetch: run both search legs independently, filtered
prefetch = [
    Prefetch(query=dense_vector, using="dense", limit=150, filter=parsed_filter),
    Prefetch(query=sparse_vector, using="sparse", limit=150, filter=parsed_filter),
]
# Fuse with RRF (default) or DBSF
results = client.query_points(
    collection_name=COLLECTION,
    prefetch=prefetch,
    query=FusionQuery(fusion=Fusion.RRF),
    limit=top_k,
    with_payload=True,
)
```

Field-aware weighting: assign `score_threshold` and `using` weights per field — lexical matches on `title` weighted at 0.5, semantic matches on `review_notes` at 0.3, sparse matches on `tags/entities` at 0.2.

### Tasks

- [ ] `src/retriever.py` — `HybridRetriever` class
  - `retrieve(parsed_query: ParsedQuery, top_k: int = 20) -> list[ScoredCandidate]`
  - Embeds `semantic_query` into both dense + sparse vectors via `Embedder`
  - Executes prefetch + RRF fusion query
  - Falls back to dense-only if sparse index is empty
  - Returns `ScoredCandidate(item: MediaItem, rrf_score: float, dense_score: float, sparse_score: float)`
- [ ] Configurable fusion method: `FUSION = "rrf"` or `"dbsf"` in config
- [ ] Logging: log filter applied, candidate counts per leg, fusion scores for top-5

### Acceptance criteria

- A query for `"mecha anime"` returns items with `type=anime` and mecha-related tags ahead of unrelated items
- A filter for `watch_status != "watched"` strictly excludes watched items from all results
- Retrieval completes in under 500ms for a 10,000-item collection

---

## Phase 4 — Score Enhancement & Recommendation Value

**Goal**: Transform the raw RRF score into a final **Recommendation Value** using business logic decay functions.

### Formula

```
RecommendationValue = rrf_score × rating_boost × recency_decay × length_decay
```

**rating_boost** — logarithmic amplifier on normalized rating:
```
rating_boost = 1 + log(1 + normalized_rating)   # normalized_rating = rating / 10
```
A 10/10 item gets ≈1.69×, a 5/10 item gets ≈1.41×, a 0/10 item gets 1.0×.

**recency_decay** — exponential decay on `year_released`, origin = current year:
```
recency_decay = exp(-λ × (current_year - year_released))   # λ = 0.05 default
```
Items from this year get 1.0; items from 20 years ago get ≈0.37. Configurable `λ`.

**length_decay** — Gaussian decay on `num_episodes_or_pages`, only applied when the user's query expresses a length preference (e.g., "short series", "under 26 episodes"):
```
# If query parser extracts a length preference:
length_decay = exp(-0.5 × ((episodes - origin) / scale)²)
# Otherwise: length_decay = 1.0 (no penalty)
```

### Tasks

- [ ] `src/scorer.py` — `Scorer` class
  - `score(candidates: list[ScoredCandidate], query_context: QueryContext) -> list[RankedResult]`
  - Applies all three decay functions
  - `QueryContext` carries `length_preference: int | None` extracted by the query parser
  - Returns sorted `RankedResult(item, recommendation_value, component_scores)` list
- [ ] Expose decay parameters in config (`lambda_recency`, `length_origin`, `length_scale`)
- [ ] `src/query_parser.py` — extend parsed output to include `length_preference_episodes: int | None`

### Acceptance criteria

- A 10/10 anime from 2023 consistently outranks a 6/10 anime from 2001 for equivalent semantic relevance
- A query for "short series" penalizes 100+ episode shows without excluding them entirely
- `recommendation_value` is always in `[0, ∞)` and monotonically reflects query relevance × quality × recency

---

## Phase 5 — Explainability (XAI)

**Goal**: Generate a structured, metadata-grounded explanation for each top-K recommendation.

### Reason generation prompt structure

The LLM receives:
1. The user's original query verbatim
2. The item's full metadata (all scalar fields + genres + tags + entities + a 300-char snippet of review_notes)
3. The item's `recommendation_value` and component scores
4. Instructions to identify intersecting concepts, flag contrastive matches (e.g., high rating compensating for partial tag mismatch), and output strict JSON

### Output schema

```json
{
  "reasons": [
    "Matches your 'psychological thriller' query via the 'psychological horror' and 'unreliable narrator' tags.",
    "Rated 9.0/10 — substantially above your query's implied quality threshold.",
    "Directed by Satoshi Kon, an entity associated with other highly-rated items in your library."
  ],
  "matched_tags": ["psychological horror", "unreliable narrator", "surrealism"]
}
```

### Tasks

- [ ] `src/explainer.py` — `Explainer` class
  - `explain_batch(results: list[RankedResult], user_query: str) -> list[ExplainedResult]`
  - Uses async Claude API calls with `asyncio.gather` — all top-K explanations run in parallel
  - System prompt enforces: no plot hallucination, cite only provided metadata, 2–4 reasons max, `matched_tags` must be a subset of the item's actual tags
  - Parses Claude's JSON output with Pydantic; falls back to a template-based reason if JSON parse fails
- [ ] Anti-hallucination guard: `matched_tags` validated against `item.tags + item.genres` at output time; any tag not present is stripped with a warning log
- [ ] `src/schema.py` — add `ExplainedResult(RankedResult)` with `reasons: list[str]` and `matched_tags: list[str]`

### Acceptance criteria

- All `matched_tags` in output are provably present in the source item's metadata
- Parallel API calls complete in under 3 seconds for top-10 results
- Fallback template produces a valid (if generic) reason on Claude API failure

---

## Phase 6 — CLI & Output Layer

**Goal**: A usable command-line interface that ties all stages together.

### Commands

```
# Ingest a dataset
recommend ingest --input data/my_library.json [--batch-size 32]

# Query
recommend query "psychological thriller anime I haven't seen, rated above 8" [--top-k 10] [--format table|json]

# Export collection back to JSON
recommend export --output data/backup.json

# Collection stats
recommend info
```

### Output (table format, default)

```
 #  Title                    Type   Year  Rating  RecVal  Reasons
─────────────────────────────────────────────────────────────────────────
 1  Perfect Blue             anime  1997    9.0    0.847   Matches 'psychological horror'...
 2  Paranoia Agent           anime  2004    8.5    0.791   Directed by Satoshi Kon...
    [web] https://...  [file] /media/anime/perfect_blue.mkv
```

### Tasks

- [ ] `src/cli.py` — Click group with `ingest`, `query`, `export`, `info` commands
- [ ] `src/pipeline.py` — `RecommendationPipeline` that wires `QueryParser → HybridRetriever → Scorer → Explainer`
- [ ] `src/output.py` — rich table renderer + JSON serializer for `list[ExplainedResult]`
- [ ] End-to-end smoke test: ingest `data/sample.json`, run 3 queries, assert non-empty results

### Acceptance criteria

- `recommend query "cyberpunk anime"` returns results in under 5 seconds on first run (cold LLM call)
- `--format json` output is valid JSON parseable by `json.loads`
- Both `local_file_location` and `web_link` appear in output when present

---

## Phase 7 — Multi-Modal Expansion

**Goal**: Extend the engine to handle books, manga, and papers without breaking the video pipeline.

### Schema changes

The core schema is already media-agnostic. The following additions are needed:

| New type | New/renamed fields | Notes |
|---|---|---|
| `book` | `num_episodes_or_pages` → pages | Already in schema; just semantics |
| `manga` | same as book + `volumes: int` (optional) | Add optional `volumes` payload |
| `paper` | `abstract: str` replaces `review_notes` as primary dense field | Query parser auto-maps |
| `image` / `art` | `description: str` → dense field; no episodes | Long-term; requires visual embedding path |

### Tasks

- [ ] `src/schema.py` — add `volumes: int | None`; add `abstract: str | None`; embedder falls back `review_notes → abstract` for papers
- [ ] `src/embedder.py` — `_get_dense_text(item)` dispatch: uses `abstract` if present and `review_notes` is empty
- [ ] `src/query_parser.py` — extend few-shot examples for book/manga/paper queries ("manga with over 30 volumes", "CS paper about attention mechanisms")
- [ ] `src/explainer.py` — extend system prompt to handle modality-specific language ("read" vs "watch", "pages" vs "episodes")
- [ ] `data/sample_books.json` — 5 book entries for smoke testing
- [ ] Verify existing video queries still pass after schema extension

### Acceptance criteria

- A mixed library (videos + books) can be ingested in one pass
- `recommend query "manga over 20 volumes"` correctly filters by `type=manga` AND `num_episodes_or_pages > 20`
- Explanation text uses "read" for books and "watch" for videos

---

## Phase 8 — Reranking & Evaluation (Advanced)

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

### Evaluation

Build a **golden query set**: 20–30 (query, expected_top_items) pairs hand-labelled from the actual library.

Metrics:
- **NDCG@10** — primary ranking quality metric; tune RRF `k` constant and decay `λ` to maximize
- **Precision@5** — fraction of top-5 results that are genuinely relevant
- **MAE** — mean absolute error between `recommendation_value` and post-consumption actual rating (requires usage logging)

### Tasks

- [ ] `src/reranker.py` — `Reranker` class using `FlagEmbedding.FlagReranker` with `bge-reranker-v2-m3`
  - `rerank(query: str, candidates: list[ScoredCandidate], top_n: int = 20) -> list[ScoredCandidate]`
- [ ] `src/pipeline.py` — add optional `--rerank` flag; insert reranker stage between retriever and scorer
- [ ] `tests/golden_queries.json` — 20 labelled query/result pairs
- [ ] `scripts/evaluate.py` — compute NDCG@10 and Precision@5 against golden set; print comparison table
- [ ] Hyperparameter sweep over `λ_recency` ∈ [0.02, 0.05, 0.1] and fusion method (RRF vs DBSF), report best NDCG@10

### Acceptance criteria

- NDCG@10 improves by ≥5% with reranker vs without on the golden set
- Evaluation script runs in under 2 minutes for 20 queries

---

## Dependency Map

```
Phase 0 (Foundation)
  └── Phase 1 (Ingestion)
        ├── Phase 2 (Query Parser)
        │     └── Phase 3 (Hybrid Retrieval)
        │           └── Phase 4 (Scoring)
        │                 └── Phase 5 (Explainability)
        │                       └── Phase 6 (CLI)
        │                             └── Phase 7 (Multi-modal)
        │                                   └── Phase 8 (Reranking)
        └── Phase 8 also depends on Phase 6 being stable
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

## File Structure (target state after Phase 6)

```
Recommendation-Engine/
├── src/
│   ├── schema.py        # Pydantic models: MediaItem, ParsedQuery, RankedResult, ExplainedResult
│   ├── config.py        # Settings from .env
│   ├── embedder.py      # BGE-M3 wrapper
│   ├── store.py         # Qdrant collection management
│   ├── ingest.py        # Ingestion CLI entry point
│   ├── query_parser.py  # LLM-based self-querying retriever
│   ├── cache.py         # LRU query cache
│   ├── retriever.py     # Hybrid search (dense + sparse + RRF)
│   ├── scorer.py        # Decay functions → Recommendation Value
│   ├── explainer.py     # Async LLM explanation generator
│   ├── pipeline.py      # End-to-end orchestration
│   ├── output.py        # Rich table + JSON renderers
│   └── cli.py           # Click commands
├── data/
│   ├── sample.json
│   └── sample_books.json
├── tests/
│   ├── test_query_parser.py
│   ├── test_scorer.py
│   └── golden_queries.json   # Phase 8
├── scripts/
│   └── evaluate.py           # Phase 8
├── reports/
│   └── Building a Smart Recommendation Engine.md
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── ROADMAP.md
```

---

## Open Questions / Future Work

- **Visual embeddings for images/art**: Phase 7 leaves images as a stub. The correct path is CLIP or VLM2Vec-V2 (which supports text + image + video in a unified embedding space) for items where the primary content is visual rather than textual.
- **Cross-item collaborative signals**: The current engine is content-based only. If the library grows large and rating patterns emerge across many items, a lightweight matrix factorization layer could augment the content-based scores — but this is only worthwhile with 500+ rated items.
- **Watch history as implicit feedback**: `watch_status` currently acts only as a filter. A future enhancement could use it as an implicit preference signal (e.g., auto-boost items similar to highly-rated watched content).
- **Web UI**: A minimal FastAPI + HTMX frontend would make the engine accessible without a terminal. Out of scope for initial phases.
