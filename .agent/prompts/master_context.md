# Master Context Prompt

```
You are working on the CRE: a local-first, hybrid semantic-lexical recommendation
engine for personal media libraries, consumed as a submodule by Image-Toolkit.

Pipeline: prompt → LLM self-querying parser ({semantic_query, SQL WHERE}) → hybrid retrieval
(dense cosine + sparse dot over a SQLite store) → RRF/DBSF fusion + decay → Recommendation Value →
LLM explainer → CLI/JSON output.

Structure: src/data (ingest/embed/export/store), src/search (parser/retriever/scorer/reranker/
explainer/pipeline), src/cli (Click `recommend` + rich output), src/core (config/cache/schema).
Roadmaps in moon/; docs in docs/; CONTRIBUTING + codecov in git/; tools/ holds the justfiles.

Rules of engagement:
- All tunables come from src/core/config.py (pydantic-settings) — no magic numbers.
- LLM (anthropic) is optional: guard it, degrade gracefully, read keys from env.
- Mock the LLM and embedder in tests; keep tests headless/offline.
- Run `just lint`, `just typecheck`, and `just test::test` before declaring done.
- Read docs/ARCHITECTURE.md before structural changes; update moon/ + docs/CHANGELOG.md after.
```
