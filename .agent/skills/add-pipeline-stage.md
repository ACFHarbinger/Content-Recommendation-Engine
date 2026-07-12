# Skill: Add / Modify a Search Pipeline Stage

Change the online query pipeline in `src/search/` (parser → retriever → scorer → reranker →
explainer, orchestrated by `pipeline.py`).

1. **Contract**: a stage takes typed input (from `src/core/schema.py`) and returns typed output;
   wire it into `pipeline.py` in the right order.
2. **Config-driven**: any weight/threshold/cap is a field in `src/core/config.py` — no literals.
3. **LLM-optional**: if the stage calls Anthropic, provide a deterministic fallback and guard the
   key check.
4. **Async**: match the surrounding stage's sync/async style; don't block the loop.
5. **Tests**: deterministic inputs → deterministic output; mock LLM/embedder; cover empty/tie cases.
6. **Docs**: update `moon/roadmaps/search.md`, `docs/ARCHITECTURE.md` if the flow changed, CHANGELOG.
