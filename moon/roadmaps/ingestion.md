# Roadmap — Ingestion & Data

The offline pipeline: load raw metadata, embed it, and persist to the SQLite store. Implementation
in [`src/data/`](../../src/data/) (`ingest.py`, `embedder.py`, `export.py`, `store.py`).

## §1 — Ingestion (done ✅, hardening)

- [x] Load media metadata (JSON/CSV) into the item schema (`src/core/schema.py`).
- [ ] Idempotent re-ingest (upsert by stable `id`); document the dedup key.
- [ ] Validation errors surface the offending record, not a stack trace.

## §2 — Embeddings

- [x] BGE-M3 dense + sparse vectors (review/notes semantic space; title/tags/genres/entities lexical space).
- [ ] Batch + cache embeddings; document VRAM/CPU footprint and the offline/online split.

## §3 — Store (SQLite)

- [x] SQLite `items` table (scalar columns + JSON array columns + vector blobs).
- [ ] Migration helper for schema changes; document the blob layout for dense/sparse vectors.
- [ ] `export.py` round-trips the store to JSON for backup/interchange.

## §4 — Future

- [ ] Optional external vector backend (pgvector/Qdrant) behind the store interface — only if a
      measured SQLite bottleneck appears.
