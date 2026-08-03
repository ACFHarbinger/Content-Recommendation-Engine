# Pull Request

## Summary

<!-- What does this PR change and why? Link the roadmap item (moon/ROADMAP.md or moon/roadmaps/). -->

## Affected Area(s)

- [ ] Ingestion & data (`src/data`)
- [ ] Search & retrieval (`src/search`)
- [ ] CLI & output (`src/cli`)
- [ ] Core & infrastructure (`src/core`)
- [ ] Docs / tooling / CI

## Type of Change

- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] ♻️ Refactor
- [ ] ⚡ Performance
- [ ] 📚 Documentation
- [ ] 🔧 Tooling / CI

## Checklist

- [ ] `just lint`, `just typecheck`, and `just test` pass
- [ ] LLM (`anthropic`) and embedder calls are mocked in tests (no network)
- [ ] Config lives in `src/core/config.py`; no hard-coded constants or secrets
- [ ] Public APIs have type annotations + Google-style docstrings
- [ ] Docs / roadmaps / `docs/CHANGELOG.md` updated where the public surface changed
