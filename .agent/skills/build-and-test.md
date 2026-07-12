# Skill: Build & Test

Standard verification sequence — run before declaring a task done.

```bash
just sync           # uv sync + editable install (first time / after dep changes)
just lint           # ruff check + ruff format --check + mypy
just typecheck      # mypy src
just test::test     # pytest (or: just test-run)
just coverage       # pytest --cov=src
```

Selective:

```bash
uv run pytest tests/test_pipeline.py -q     # one module
uv run recommend query "cyberpunk anime" --json   # smoke-test the CLI
just evaluate                                # retrieval quality (NDCG@K, P@5) — no LLM
```

Tests are offline: the LLM (`anthropic`) and embedder are mocked via `tests/conftest.py`.
