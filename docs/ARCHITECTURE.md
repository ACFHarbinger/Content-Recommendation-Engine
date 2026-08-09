# Architecture

Local-first, hybrid semantic-lexical recommendation engine for personal media libraries. This is a
concise blueprint; the phased build history and per-module detail live in
[`moon/ROADMAP.md`](../moon/ROADMAP.md) and [`moon/roadmaps/`](../moon/roadmaps/).

## Pipeline

A five-stage pipeline — stages 1–2 run **offline** (ingestion), stages 3–5 run **online** (query time).

```
[Offline]  Raw metadata (JSON/CSV)
             → BGE-M3 embeddings (dense = semantic, sparse = lexical)
             → SQLite store (scalar + JSON-array columns + vector blobs)

[Online]   User prompt
             1. Self-Querying Retriever (LLM) → {semantic_query, SQL WHERE}
             2. Hybrid Retrieval             → cosine(dense) + dot(sparse), filtered by WHERE
             3. Score Fusion (RRF / DBSF)    → + business-logic decay → Recommendation Value
             4. LLM Explainer                → {reasons[], matched_tags[]} per top-K
             5. Output                       → CLI (rich) / JSON
```

## Module Map

| Path | Layer | Responsibility |
| :--- | :--- | :--- |
| `src/data/` | Ingestion | `ingest`, `embedder` (BGE-M3), `export`, `store` (SQLite) |
| `src/search/` | Retrieval | `query_parser`, `retriever`, `scorer`, `reranker`, `explainer`, `pipeline` |
| `src/cli/` | Interface | `cli` (Click, entry point `recommend`), `output` (rich / JSON) |
| `src/core/` | Infra | `config` (pydantic-settings), `cache`, `schema` |

## Key Design Decisions

- **Local-first**: SQLite (stdlib) is the vector store — no external services required
  (`infra/global/docker/docker-compose.yml` is a placeholder for a possible future backend).
- **Hybrid retrieval**: dense (semantic) + sparse (lexical) vectors fused with RRF (default) or DBSF.
- **LLM-optional**: Anthropic Claude powers the self-querying parser and the explainer; the engine
  degrades to deterministic behaviour without an API key (roadmap).
- **Explainability first**: every result carries reasons and matched tags derived from item fields.

## Boundaries

The engine is consumed as a submodule by **Image-Toolkit**; the stable contract is the `recommend`
CLI entry point and its JSON output. See [`moon/roadmaps/cli.md`](../moon/roadmaps/cli.md).
