<div align="center">

# Recommendation-Engine

**A local-first, hybrid semantic-lexical search and AI-powered recommendation engine for personal media libraries.**

<a href="https://github.com/ACFHarbinger/Image-Toolkit/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ACFHarbinger/Image-Toolkit/actions/workflows/ci.yml/badge.svg"></a>
<img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
<a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>

</br>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white"></a>
<a href="https://qdrant.tech/"><img alt="Qdrant" src="https://img.shields.io/badge/Qdrant-Vector_Search-FF4154?logo=qdrant&logoColor=white"></a>
<a href="https://www.anthropic.com/"><img alt="Anthropic Claude" src="https://img.shields.io/badge/Claude-AI_Query_Parser-191919?logo=anthropic&logoColor=white"></a>
<a href="https://docs.pydantic.dev/"><img alt="Pydantic" src="https://img.shields.io/badge/Pydantic-Data_Validation-E23B7E?logo=pydantic&logoColor=white"></a>
<a href="https://docs.pytest.org/"><img alt="pytest" src="https://img.shields.io/badge/pytest-testing-0A9EDC?logo=pytest&logoColor=white"></a>

</br>

<a href="https://github.com/astral-sh/uv"><img alt="uv" src="https://img.shields.io/badge/managed%20by-uv-261230.svg"></a>
<a href="https://www.docker.com/"><img alt="Docker" src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white"></a>
<a href="https://dependabot.com/"><img alt="Dependabot" src="https://img.shields.io/badge/Dependabot-enabled-025E8C?logo=dependabot&logoColor=white"></a>

<p>
  <a href="#-quick-start"><strong>🚀 Quick Start</strong></a> |
  <a href="#-features"><strong>✨ Features</strong></a> |
  <a href="#%EF%B8%8F-how-it-works"><strong>⚙️ How It Works</strong></a> |
  <a href="#-installation--setup"><strong>📦 Installation</strong></a> |
  <a href="#-cli-commands"><strong>💻 CLI Commands</strong></a> |
  <a href="#-data-format"><strong>📄 Data Format</strong></a> |
  <a href="#-configuration"><strong>🔧 Configuration</strong></a> |
  <a href="#-development"><strong>🛠️ Development</strong></a> |
  <a href="#-architecture-decisions"><strong>📚 Architecture Decisions</strong></a>
</p>

</div>

---

## ⚙️ How It Works

The engine combines local machine learning models with optional cloud LLM APIs to parse natural language queries, retrieve candidates, score them, and explain the recommendations:

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

> [!NOTE]
> All ML models run completely locally. Only the query parser and explainer invoke the Anthropic API. Both are optional — the engine works out-of-the-box without an API key using semantic-only search and template fallbacks.

---

## ✨ Features

- 🧠 **Hybrid Search** - Combines sparse lexical matches with dense semantic vector searches.
- 🐳 **Qdrant Vector DB** - High-performance candidate retrieval with direct metadata filtering.
- ⚡ **Multiplicative Scoring** - Multi-factor ranking utilizing item rating, recency decay, duration penalty, and watch history.
- 🤖 **AI Query Parser & Explainer** - Intelligent intent parsing and automated query explanations using Claude.
- 📦 **Local-First & Portable** - Runs completely locally on-disk or via Docker Compose.
- 💻 **Interactive CLI** - Styled rich terminal tables and JSON output formats.

---

## 🚀 Quick Start

### 1. Clone and Install
```bash
git clone <repo-url>
cd Recommendation-Engine
pip install -e ".[dev]"
```

### 2. Start Local Qdrant (Docker)
```bash
docker compose -f infra/global/docker/docker-compose.yml up -d
```
The Qdrant UI dashboard will be available at [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

> [!TIP]
> If you don't have Docker installed, the engine will automatically default to local directory storage (set `QDRANT_LOCAL_PATH=.qdrant_data` or leave it empty).

### 3. Configure
```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY for parser/explanations
```

### 4. Ingest Sample Library
```bash
# Ingest the included sample libraries
recommend ingest --input data/sample.json
recommend ingest --input data/sample_books.json
```

### 5. Run Queries
```bash
# Natural language queries
recommend query "psychological thriller anime I haven't watched"

# Highly filtered queries
recommend query "highly rated sci-fi books from the 80s"

# Output in JSON format
recommend query "space opera" --format json

# High-precision retrieval with cross-encoder reranker
recommend query "dark philosophical themes" --rerank
```

---

## 📦 Installation & Setup

### Prerequisites
- **Python** (v3.11+)
- **Docker & Docker Compose** (Optional, for Qdrant container)

### Standard Installation
Install the project in editable mode with development dependencies:
```bash
pip install -e ".[dev]"
```

---

## 💻 CLI Commands

The command-line interface provides the following subcommands:

| Subcommand | Description |
|:---|:---|
| `recommend ingest --input FILE` | Embed and upsert a JSON library into Qdrant |
| `recommend sync --input FILE` | Incrementally update existing items by UUID |
| `recommend export --output FILE` | Export the Qdrant collection back to JSON |
| `recommend query TEXT` | Run the full recommendation pipeline |
| `recommend delete UUID` | Remove a single item from the collection by UUID |
| `recommend info` | Show collection stats and configuration |

### Query CLI Flags

Customize your recommendation queries using the following flags:

| Flag | Default | Effect |
|:---|:---|:---|
| `--top-k N` | `10` | Number of results to return |
| `--format table\|json` | `table` | Output presentation format |
| `--rerank` | `off` | Enable cross-encoder reranking |
| `--no-explain` | `off` | Skip LLM explanation generation (faster) |
| `--no-history` | `off` | Disable history-based query boosts |
| `-v / --verbose` | `off` | Show detailed debug logging |

---

## 📄 Data Format

Items ingested into the library are standard JSON objects. The only required fields are `id` and `title`.

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

### Valid Values & Guidelines
- **Type values**: `anime`, `show`, `movie`, `book`, `manga`, `paper`, `game`, `other`
- **Status values**: `watched`, `reading`, `plan_to_watch`, `on_hold`, `dropped`
- **Genres/tags**: Comma-separated string or JSON array
- **Paper-specific**: Use the `abstract` field instead of `review` as the primary dense embedding source.
- **Manga-specific**: Include `volumes: int`.

---

## 🔧 Configuration

Configure your engine parameters in the `.env` file:

| Variable | Default | Description |
|:---|:---|:---|
| `ANTHROPIC_API_KEY` | — | API key for Claude query parser and explainer |
| `QDRANT_URL` | — | Remote Qdrant server URL (leave blank for local files) |
| `QDRANT_LOCAL_PATH` | `.qdrant_data` | Storage path when using local file-system mode |
| `QDRANT_COLLECTION` | `listings` | Target collection name |
| `EMBED_MODEL` | `BAAI/bge-m3` | Vector embedding model (~2 GB first-run download) |
| `DEFAULT_TOP_K` | `10` | Default result candidate pool size |
| `LAMBDA_RECENCY` | `0.05` | Recency decay rate (0 disables recency decay) |
| `LENGTH_SCALE` | `24` | Gaussian variance scaling for episode-length preferences |
| `FUSION_METHOD` | `rrf` | Candidate score fusion: `rrf` (Rank Reciprocal Fusion) or `dbsf` |
| `HISTORY_MIN_RATING` | `7.0` | Minimum rating to trigger watch-history boosting |
| `HISTORY_BOOST_WEIGHT` | `0.15` | Strength of watch-history similarity signal (0-1) |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3`| Cross-encoder model used when `--rerank` is enabled |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model selection for parsing and explanations |

---

## 🛠️ Development

Run the automated test suite and optimization scripts:

```bash
# Run tests (no external dependencies or APIs needed)
pytest

# Evaluate retrieval quality against golden queries
python scripts/evaluate.py --k 10

# Hyperparameter grid search sweep (requires populated Qdrant)
python scripts/sweep.py --k 10 --csv sweep_results.csv
```

### Test Suite Structure

| Module | Tests |
|:---|:---|
| schema | 12 |
| scorer | 24 |
| query_parser | 29 |
| store | 13 |
| retriever | 10 |
| explainer | 8 |
| pipeline | 8 |
| output | 11 |
| **Total** | **115** |

> [!TIP]
> All unit tests are mocked and execute offline without Qdrant, BGE-M3, or Anthropic API dependencies.

---

## 📚 Architecture Decisions

- **Why BGE-M3?** A single call produces dense, sparse, and ColBERT vectors simultaneously, avoiding embedding space mismatches between retrieval steps.
- **Why Qdrant?** The ACORN algorithm integrates metadata filtering directly into HNSW graph traversal. This is crucial for personal libraries where a large percentage of items (e.g. `watch_status != "watched"`) are filtered out.
- **Why RRF over DBSF?** Rank Reciprocal Fusion is distribution-agnostic and robust when one retrieval step has weak scoring. DBSF (`FUSION_METHOD=dbsf`) can be used if exact title matching is preferred.
- **Why Multiplicative Scoring?** A zero-relevance item stays near zero regardless of high ratings or recency. Additive formulas frequently float highly rated but irrelevant items to the top.
- **Why Async Explanations?** Generating 10 Claude explanations sequentially takes 10-30s. Using `asyncio.gather` parallelizes requests and keeps response latency under 3 seconds.

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
