# Workflow — Feature Implementation

1. **Scope**: identify the stage (data / search / cli / core) + its `moon/roadmaps/` item; read
   `docs/ARCHITECTURE.md` if the pipeline flow changes.
2. **Schema/config first**: add schema fields (`src/core/schema.py`) and tunables
   (`src/core/config.py`) before the code depends on them.
3. **Implement** per the relevant skill; keep LLM/network use guarded and optional.
4. **Test**: deterministic unit tests with LLM/embedder mocked; cover edge cases.
5. **Gate**: `just lint && just typecheck && just test::test`.
6. **Document**: roadmap tick, `docs/ARCHITECTURE.md` if flow changed, `docs/CHANGELOG.md`.
