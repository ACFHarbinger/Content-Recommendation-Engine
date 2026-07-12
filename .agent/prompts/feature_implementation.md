# Feature Implementation Prompt

```
Implement the following Recommendation-Engine feature: {FEATURE}.

Before writing code:
1. Locate the pipeline stage (data / search / cli / core) and read the matching moon/roadmaps/ file
   and docs/ARCHITECTURE.md.
2. Check src/core/schema.py + config.py — new fields/tunables belong there.

Requirements:
- Any new tunable (weight, decay, cap, path) is a config field in src/core/config.py — not a
  literal in business logic.
- Guard LLM/network use so the engine still works offline / without ANTHROPIC_API_KEY.
- Type annotations + Google-style docstrings; async where the stage is async.
- Add pytest coverage with the LLM + embedder mocked (fixtures in tests/conftest.py).
- Update: the module roadmap in moon/roadmaps/, docs/ARCHITECTURE.md if the pipeline changed,
  and docs/CHANGELOG.md.
```
