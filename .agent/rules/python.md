# Rules — Python

- Python 3.11+; type annotations + Google-style docstrings on public APIs; prefer precise types
  over `Any`.
- Ruff is the formatter and linter (`line-length = 120`); mypy (permissive baseline, tightened
  per-module). `just lint` mirrors CI (`ruff check` + `ruff format --check`).
- **All tunables live in `src/core/config.py`** (pydantic-settings) — no scattered constants or
  magic numbers in business logic.
- The item schema (`src/core/schema.py`) is the single source of truth for record shape; validate
  with pydantic and surface the offending record on error.
- Secrets/keys come from the environment (`.env` / pydantic-settings) — never hard-coded.
- Async stages (explainer, pipeline) must not block; test with `pytest-asyncio`.
- Every bug fix ships a regression test.
