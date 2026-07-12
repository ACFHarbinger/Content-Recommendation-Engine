# Rules — Test Writing

- Framework: pytest (+ pytest-asyncio, pytest-mock) under `tests/`. Tests are headless, fast, and
  **offline** — no real LLM or network calls.
- Use the shared fixtures (`tests/conftest.py`): `MockEmbedder`, a temp-file `sqlite_store`, and a
  `cfg` on `tmp_path`. Never hit a real SQLite path or a real Anthropic endpoint.
- Mock `anthropic` responses for parser/explainer tests; assert on structure (parsed filter shape,
  reasons cite real tags) not exact text.
- Retrieval/scoring: deterministic inputs → deterministic ranking; cover empty-candidate and
  tie-break edge cases.
- Golden queries (`tests/golden_queries.json`) back the evaluation harness; grow the set with each
  real failure.
- Coverage target: [`git/codecov.yaml`](../../git/codecov.yaml).
