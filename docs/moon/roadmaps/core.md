# Roadmap — Core & Infrastructure

Configuration, caching, schema, and cross-cutting quality/tooling. Implementation in
[`src/core/`](../../src/core/) (`config.py`, `cache.py`, `schema.py`).

## §1 — Config & schema

- [x] Pydantic settings (`config.py`) + item schema (`schema.py`).
- [ ] Single source of truth for tunables (fusion weights, decay, candidate caps); documented defaults.
- [ ] `.env.example` covers every setting the engine reads.

## §2 — Caching

- [x] Cache layer (`cache.py`) for embeddings / parsed queries.
- [ ] Document cache invalidation + on-disk location.

## §3 — Quality & tooling

- [ ] Fix `[tool.ruff] src` / `[tool.mypy] overrides` left over from the Image-Toolkit template
      (they reference `backend/src`, not this repo's `src`).
- [ ] Raise mypy strictness per-module as annotations land; `just lint` + `just test` green in CI.
- [ ] Coverage target tracked in [`git/codecov.yaml`](../../git/codecov.yaml).
