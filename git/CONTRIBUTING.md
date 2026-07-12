# Contributing to Recommendation-Engine

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Thanks for contributing! This is the local-first hybrid recommendation engine consumed as a
submodule by **Image-Toolkit**. AI coding assistants should also read
[`.agent/AGENTS.md`](../.agent/AGENTS.md).

## 1. Getting Started

- Read [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the pipeline and module map, and the
  top-level [README](../README.md) for setup and CLI usage.
- Most work should advance an item in [`moon/ROADMAP.md`](../moon/ROADMAP.md) or a module roadmap
  under [`moon/roadmaps/`](../moon/roadmaps/).

## 2. Development Setup

```bash
uv sync                 # install runtime + dev dependencies
uv pip install -e ".[dev]"
just hooks              # install pre-commit hooks (optional)
```

Requires Python 3.11+. An `ANTHROPIC_API_KEY` (see `.env.example`) enables the LLM query parser and
explainer; the engine should degrade gracefully without one.

## 3. Code Style

- **Ruff** is the formatter and linter (`line-length = 120`); **mypy** for typing (permissive
  baseline, tightened per-module). Run `just lint` and `just typecheck`.
- Public functions/classes get type annotations and Google-style docstrings; prefer precise types
  over `Any`.
- Configuration comes from `src/core/config.py` (pydantic-settings) — no scattered constants.
- Never hard-code secrets; read keys from the environment (`.env`).

## 4. Git Workflow

- Branch from `main`: `feature/<slug>`, `fix/<slug>`, `docs/<slug>`.
- Conventional-style commit subjects (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`); explain the
  *why* in the body and reference the roadmap item.

## 5. Testing

```bash
just test               # pytest (uses pytest-asyncio / pytest-mock)
```

- Keep tests headless and fast — mock the LLM (`anthropic`) and embedder; no network.
- Every bug fix ships a regression test. Coverage target: see [`codecov.yaml`](codecov.yaml).

## 6. Pull Requests

1. `just lint`, `just typecheck`, and `just test` pass locally.
2. Fill in the [PR template](../.github/PULL_REQUEST_TEMPLATE.md).
3. Update docs / roadmaps / `docs/CHANGELOG.md` for any public-surface change.
4. CI (ruff + pytest) must be green.

## 7. Adding Components

See the skill guides in [`.agent/skills/`](../.agent/skills/) — adding a pipeline stage, a data
source, or a CLI command.
