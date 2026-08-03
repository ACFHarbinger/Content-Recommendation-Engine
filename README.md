# Recommendation Engine

A personal, local-first media recommendation engine.  
Takes your library of anime, movies, books, and other media — annotated with ratings, genres, and tags — and returns ranked recommendations with explanations, using hybrid semantic + lexical search.

---

## How it works

```
Your query (natural language + optional filters)
       │
       ▼
  QueryParser (Claude API)  ─→  {semantic_query, filters}
       │
       ▼
  HybridRetriever (Qdrant)
    ├── Dense k-NN on review/notes embeddings  ─┐
    └── Sparse k-NN on title/tag embeddings    ─┘
       │            RRF fusion
       ▼
  [Reranker]  (optional, bge-reranker-v2-m3)
       │
       ▼
  Scorer  →  rating_boost × recency_decay × length_decay × history_boost
       │
       ▼
  Explainer (Claude API)  →  per-result reasons + matched tags
       │
       ▼
  Rich table / JSON output
```

All ML runs locally. Only the query parser and explainer call the Anthropic API (both optional — the engine works without an API key using semantic-only search and template fallbacks).

---

## Quickstart

### 1 — Clone and install

```bash
git clone <repo>
cd Recommendation-Engine
pip install -e ".[dev]"
```

### 2 — Start local Qdrant (Docker)

```bash
docker compose up -d
```

Qdrant UI is available at http://localhost:6333/dashboard.

> **No Docker?** The engine also stores vectors in a local directory by default  
> (set `QDRANT_LOCAL_PATH=.qdrant_data` or leave it blank).  
> In that case, skip this step.

### 3 — Configure

```bash
cp .env.example .env
# Edit .env — at minimum, set ANTHROPIC_API_KEY if you want query parsing + explanations
```

### 4 — Ingest your library

```bash
# Ingest the included sample data
recommend ingest --input data/sample.json
recommend ingest --input data/sample_books.json

# Or your own library (see Data Format below)
recommend ingest --input /path/to/my_library.json
```

### 5 — Query

```bash
# Natural language query
recommend query "psychological thriller anime I haven't watched"

# With filters extracted automatically
recommend query "highly rated sci-fi books from the 80s"

# JSON output
recommend query "space opera" --format json

# Without LLM explanations (faster)
recommend query "mecha anime" --no-explain

# With cross-encoder reranker (slower, higher precision)
recommend query "dark philosophical themes" --rerank
```

---

## Commands

| Command | Description |
|---|---|
| `recommend ingest --input FILE` | Embed + upsert a JSON library into Qdrant |
| `recommend sync --input FILE` | Incrementally update existing items by UUID |
| `recommend export --output FILE` | Export the Qdrant collection back to JSON |
| `recommend query TEXT` | Run the full recommendation pipeline |
| `recommend delete UUID` | Remove a single item from the collection |
| `recommend info` | Show collection stats and current config |

### `recommend query` flags

| Flag | Default | Effect |
|---|---|---|
| `--top-k N` | 10 | Number of results |
| `--format table\|json` | table | Output format |
| `--rerank` | off | Enable cross-encoder reranker |
| `--no-explain` | off | Skip LLM explanation (faster) |
| `--no-history` | off | Disable watch-history boost |
| `-v / --verbose` | off | Show DEBUG logs |

---

## Data format

Items are JSON objects. All fields except `id` and `title` are optional.

```json
{
  "id": "uuid-v4",
  "title": "Neon Genesis Evangelion",
  "type": "anime",
  "status": "watched",
  "rating": 9.5,
  "year": 1995,
  "episodes": 26,
  "genres": "Mecha, Psychological, Drama",
  "tags": "existential, post-apocalyptic, religious-symbolism",
  "associated_entities": ["Hideaki Anno"],
  "review": "A deeply personal and psychologically intense masterwork.",
  "web_link": "https://myanimelist.net/anime/30/",
  "local_file": "/media/anime/evangelion.mkv"
}
```

**Type values**: `anime`, `show`, `movie`, `book`, `manga`, `paper`, `game`, `other`  
**Status values**: `watched`, `reading`, `plan_to_watch`, `on_hold`, `dropped`  
**Genres/tags**: comma-separated string or JSON array  
**Paper-specific**: use `abstract` instead of `review` as the primary dense embedding field  
**Manga-specific**: add `volumes: int`

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for query parsing + explanations |
| `QDRANT_URL` | — | Remote Qdrant URL (leave blank for local file mode) |
| `QDRANT_LOCAL_PATH` | `.qdrant_data` | Local storage path (when QDRANT_URL is unset) |
| `QDRANT_COLLECTION` | `listings` | Collection name |
| `EMBED_MODEL` | `BAAI/bge-m3` | Embedding model (~2 GB download on first run) |
| `DEFAULT_TOP_K` | `10` | Default number of results |
| `LAMBDA_RECENCY` | `0.05` | Recency decay rate (0 = no decay) |
| `LENGTH_SCALE` | `24` | Gaussian width for episode-length preference |
| `FUSION_METHOD` | `rrf` | Score fusion: `rrf` or `dbsf` |
| `HISTORY_MIN_RATING` | `7.0` | Minimum rating for watch-history boost |
| `HISTORY_BOOST_WEIGHT` | `0.15` | Strength of watch-history signal (0–1) |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder model for `--rerank` |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model for parser + explainer |

---

## Development

```bash
# Run tests (no external dependencies needed)
pytest

# Evaluate retrieval quality against golden queries
python scripts/evaluate.py --k 10

# Hyperparameter sweep (requires populated Qdrant)
python scripts/sweep.py --k 10 --csv sweep_results.csv
```

### Test coverage

| Module | Tests |
|---|---|
| schema | 12 |
| scorer | 24 |
| query_parser | 29 |
| store | 13 |
| retriever | 10 |
| explainer | 8 |
| pipeline | 8 |
| output | 11 |
| **Total** | **115** |

All tests run without Qdrant, BGE-M3, or the Anthropic API — external dependencies are mocked.

---

## Architecture decisions

**Why BGE-M3?** A single model call produces dense + sparse + ColBERT vectors simultaneously. No embedding space mismatch between retrieval legs.

**Why Qdrant?** The ACORN algorithm integrates metadata filtering directly into HNSW graph traversal. Essential for a personal library where 80–99% of items may be filtered (e.g. `watch_status != "watched"`).

**Why RRF over DBSF?** RRF is distribution-agnostic and robust when one retrieval leg is weak. Switch to DBSF (`FUSION_METHOD=dbsf`) when exact title matches should dominate.

**Why multiplicative scoring?** A zero-relevance item (RRF score ≈ 0) stays near zero regardless of its rating or recency. Additive formulas let a high-rated but irrelevant item float to the top.

**Why async explanation?** Running 10 Claude calls sequentially adds ~10–30 s. `asyncio.gather` reduces this to ~1–3 s (single call latency).

## License

This project is dual-licensed under an open-core model:

- **Open source (free) — GNU AGPL-3.0.** Free to use, modify, and
  distribute for hobbyists, students, researchers, non-profits, and any
  other use that complies with the [AGPL-3.0](LICENSE.md)'s copyleft and
  network source-disclosure terms.
- **Commercial (paid).** For proprietary, closed-source, or SaaS use that
  can't comply with the AGPL's obligations, a paid
  [commercial license](LICENSE.txt) is available — contact ACFHarbinger
  <afonso.fernandes100@gmail.com> for pricing and terms.
