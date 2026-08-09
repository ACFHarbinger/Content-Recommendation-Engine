# AGENTS.md — Recommendation-Engine: Coding-Assistant Handbook

Instructions for AI assistants (and humans) working on this codebase.

## 1. Overview

A **local-first, hybrid semantic-lexical** recommendation engine for personal media libraries. A
free-form prompt is parsed by an LLM into a semantic query + SQL filter, candidates are retrieved
by hybrid (dense + sparse) vector math over a SQLite store, fused/scored into a **Recommendation
Value**, and each top-K result is explained. Consumed as a submodule by **Image-Toolkit**; the
stable contract is the `recommend` CLI + its JSON output.

Design record: [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md). Roadmaps:
[`moon/ROADMAP.md`](../moon/ROADMAP.md) (master) + [`moon/roadmaps/`](../moon/roadmaps/).

## 2. Tech Stack

- **Runtime**: Python 3.11+ managed with **uv**.
- **Embeddings**: `FlagEmbedding` (BGE-M3 dense + sparse); reranker bge-reranker-v2-m3.
- **LLM**: Anthropic Claude (query parser + explainer) via `anthropic` + `langchain-core` — optional.
- **Store**: SQLite (stdlib) — items table with scalar + JSON-array columns + vector blobs.
- **CLI/UI**: `click` + `rich`. **Config**: `pydantic` / `pydantic-settings`.
- **Quality**: ruff (line-length 120), mypy (permissive, tightening); pytest + pytest-asyncio/mock.

## 3. Project Structure

```
src/
├── data/     # ingest, embedder (BGE-M3), export, store (SQLite)   ← offline
├── search/   # query_parser, retriever, scorer, reranker, explainer, pipeline   ← online
├── cli/      # cli (Click, entry point `recommend`), output (rich / JSON)
└── core/     # config (pydantic-settings), cache, schema
```

Repo layout: `moon/` (roadmaps), `docs/` (ARCHITECTURE + CHANGELOG), `git/` (CONTRIBUTING +
codecov), `tools/` (justfile sub-modules), `.github/` (CI + templates), `data/` (sample datasets),
`scripts/` (evaluate/sweep), `tests/`, `infra/global/docker/` (placeholder compose).

## 4. Common Commands (just)

| Action | Command |
| :--- | :--- |
| Sync env | `just sync` |
| Lint (CI-equiv) + mypy | `just lint` |
| Type-check | `just typecheck` |
| Tests | `just test::test` (or `just test-run`) |
| Coverage | `just coverage` |
| Ingest / query | `just ingest …` / `just query …` |
| Evaluate | `just evaluate` |
| List everything | `just help` |

## 5. Coding Standards

- Type annotations + Google-style docstrings on public APIs; prefer precise types over `Any`.
- **All tunables live in `src/core/config.py`** (pydantic-settings): fusion weights, decay factors,
  candidate caps, paths. No scattered constants or magic numbers.
- **LLM-optional**: guard `anthropic` usage so the engine degrades gracefully without an API key;
  read keys from the environment (`.env`), never hard-code.
- Async: the explainer and some pipeline stages are async — don't block the loop; use
  `pytest-asyncio` for tests.
- The item schema (`src/core/schema.py`) is the single source of truth for record shape.

## 6. Testing

- pytest under `tests/`; **mock the LLM (`anthropic`) and the embedder** — tests are headless,
  fast, and offline. Fixtures in `tests/conftest.py` (`MockEmbedder`, temp-file `sqlite_store`).
- TM/store tests use a temp-file SQLite DB. Every bug fix ships a regression test.
- Coverage target: [`git/codecov.yaml`](../git/codecov.yaml).

## 7. Documentation Discipline

- Completed roadmap items move from `moon/` to [`docs/CHANGELOG.md`](../docs/CHANGELOG.md).
- New pipeline stages get a `moon/roadmaps/<area>.md` update, a docs note, and config-driven params.
- Known cleanup: `[tool.ruff] src` / `[tool.mypy] overrides` in `pyproject.toml` still reference the
  Image-Toolkit template's `backend/src` — fix to this repo's `src` when you touch tooling.
