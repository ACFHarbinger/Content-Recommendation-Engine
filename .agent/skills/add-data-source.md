# Skill: Add a Data Source / Media Type

Extend ingestion in `src/data/` to accept a new dataset or media type.

1. **Schema**: extend `src/core/schema.py` (new type/fields) additively; keep existing records valid.
2. **Ingest**: teach `ingest.py` to load + validate the new format (JSON/CSV); surface bad records.
3. **Embed**: reuse `embedder.py` (BGE-M3 dense + sparse) — decide which fields are semantic vs
   lexical and document it.
4. **Store**: ensure `store.py` upsert handles the new columns/blobs; add a migration if the table
   shape changes.
5. **Sample data**: add a small fixture under `data/` and a loader test (no network).
6. **Docs**: `moon/roadmaps/ingestion.md`, data-format section of the README, CHANGELOG.
